"""Paso 4 del flujo: narra el guion revisado con tu voz clonada (Qwen3-TTS).

Uso:
    python3 voz.py --voz mi_voz.m4a --voz-texto mi_voz.txt
    python3 voz.py --voz mi_voz.m4a --voz-texto mi_voz.txt --solo 1   # rehace solo la escena 1

Lee salidas/guion.json (debe tener "revisado": true), genera cada escena con
Qwen3-TTS y la vuelve a escuchar con Whisper: si entiende mal más del 25 % de
las palabras, la genera de nuevo. Escribe salidas/reel.wav y salidas/escenas.json
(tiempos de cada escena, para el video).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata

try:
    import mlx_whisper
except ModuleNotFoundError:
    # Si se ejecuta con el python3 del sistema, se relanza con el de .venv
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python, *sys.argv])
    sys.exit("Falta el entorno: uv pip install --python .venv/bin/python mlx-audio mlx-whisper jiwer num2words")

import jiwer
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model
from num2words import num2words

MODELO_TTS = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
MODELO_ASR = "mlx-community/whisper-large-v3-turbo"
WER_MAX = 0.25
INTENTOS = 3
PAUSA = 0.3  # segundos de silencio entre escenas
SILENCIO_INICIO = 0.5  # segundos antes de la primera escena
PRIMERA_PROB_MIN = 0.9  # Qwen3-TTS a veces se come el inicio de la primera palabra


def normalizar(texto):
    """Para comparar lo que se quiso decir con lo que Whisper oyó."""
    texto = re.sub(r"\d+(?:[.,]\d+)?", lambda m: num2words(float(m.group().replace(",", ".")) if "," in m.group()
                                                       or "." in m.group() else int(m.group()), lang="es"), texto)
    texto = unicodedata.normalize("NFC", texto.lower())
    texto = re.sub(r"ientas\b", "ientos", texto)  # num2words da "doscientos"; el guion dice "doscientas"
    texto = re.sub(r"[^\w\s]", " ", texto)
    return " ".join(texto.split())


def transcribir(ruta, palabras=False):
    r = mlx_whisper.transcribe(ruta, path_or_hf_repo=MODELO_ASR, language="es", word_timestamps=palabras)
    return (r["text"].strip(), r["segments"]) if palabras else r["text"].strip()


def recortar(audio, sr, umbral=0.02, margen=0.08, fundido=0.015):
    """Quita el silencio del inicio y del final, y suaviza la entrada y la salida."""
    ventana = int(0.01 * sr)
    energia = np.sqrt(np.convolve(audio ** 2, np.ones(ventana) / ventana, mode="same"))
    voz = np.where(energia > umbral)[0]
    if len(voz):
        audio = audio[max(0, voz[0] - int(margen * sr)):voz[-1] + int(margen * sr)]
    n = int(fundido * sr)
    audio = audio.copy()
    audio[:n] *= np.linspace(0, 1, n)
    audio[-n:] *= np.linspace(1, 0, n)
    return audio


def evaluar(ruta, frase):
    """WER de la escena y confianza de Whisper en la primera palabra."""
    oido, segmentos = transcribir(ruta, palabras=True)
    wer = jiwer.wer(normalizar(frase), normalizar(oido))
    palabras = [w for s in segmentos for w in s.get("words", [])]
    primera = palabras[0]["probability"] if palabras else 0.0
    return wer, primera, oido


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz", required=True, help="grabación de referencia (.m4a, .wav, .mp3)")
    ap.add_argument("--voz-texto", help="transcripción exacta de la grabación; si falta, la hace Whisper")
    ap.add_argument("--guion", default="salidas/guion.json")
    ap.add_argument("--salida", default="salidas")
    ap.add_argument("--tts", default=MODELO_TTS)
    ap.add_argument("--solo", help="regenera solo estas escenas (p. ej. 1,6); las demás se toman de salidas/escenas")
    args = ap.parse_args()

    guion = json.load(open(args.guion, encoding="utf-8"))
    if guion.get("revisado") is not True:
        sys.exit(f'El guion no está revisado: cambien "revisado" a true en {args.guion}')

    os.makedirs(os.path.join(args.salida, "escenas"), exist_ok=True)
    ref_wav = os.path.join(args.salida, "voz_referencia.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.voz, "-ac", "1", "-ar", "24000", ref_wav],
                   check=True)

    if args.voz_texto:
        ref_text = open(args.voz_texto, encoding="utf-8").read().strip()
    else:
        ref_text = transcribir(ref_wav)
        print(f"Transcripción de la voz (revísenla): {ref_text}")

    modelo = load_model(args.tts)
    sr = modelo.sample_rate
    solo = {int(x) for x in args.solo.split(",")} if args.solo else None
    previas = {}
    ruta_escenas = os.path.join(args.salida, "escenas.json")
    if solo and os.path.exists(ruta_escenas):
        previas = {i: e for i, e in enumerate(json.load(open(ruta_escenas, encoding="utf-8")), 1)}

    partes, escenas = [np.zeros(int(SILENCIO_INICIO * sr), dtype=np.float32)], []
    t = SILENCIO_INICIO

    for i, escena in enumerate(guion["escenas"], 1):
        frase = escena["frase"]
        ruta = os.path.join(args.salida, "escenas", f"escena_{i:02d}.wav")

        if solo and i not in solo and os.path.exists(ruta):
            audio, _ = sf.read(ruta, dtype="float32")
            audio = recortar(audio, sr)
            sf.write(ruta, audio, sr)
            wer = previas.get(i, {}).get("wer", 0.0)
        else:
            mejor = None
            for intento in range(1, INTENTOS + 1):
                audio = np.concatenate([np.array(r.audio) for r in modelo.generate(
                    text=frase, ref_audio=ref_wav, ref_text=ref_text, lang_code="spanish")]).astype(np.float32)
                audio = recortar(audio, sr)
                sf.write(ruta, audio, sr)
                wer, primera, oido = evaluar(ruta, frase)
                print(f"Escena {i} · intento {intento}: {len(audio) / sr:.1f} s · WER {wer:.0%} · "
                      f"primera palabra {primera:.0%}")
                puntaje = wer + max(0.0, PRIMERA_PROB_MIN - primera)
                if mejor is None or puntaje < mejor[0]:
                    mejor = (puntaje, wer, audio, oido)
                if wer <= WER_MAX and primera >= PRIMERA_PROB_MIN:
                    break
            _, wer, audio, oido = mejor
            if wer > WER_MAX:
                print(f"  ¡¡ Escena {i}: tras {INTENTOS} intentos Whisper oyó «{oido}»")
            sf.write(ruta, audio, sr)

        dur = len(audio) / sr
        escenas.append({**escena, "inicio": round(t, 3), "fin": round(t + dur, 3), "wer": round(wer, 3)})
        partes += [audio, np.zeros(int(PAUSA * sr), dtype=np.float32)]
        t += dur + PAUSA

    reel = np.concatenate(partes)
    sf.write(os.path.join(args.salida, "reel.wav"), reel, sr)
    json.dump(escenas, open(os.path.join(args.salida, "escenas.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n{args.salida}/reel.wav: {len(reel) / sr:.1f} s")
    if len(reel) / sr > 60:
        print("  ¡¡ Pasa de 60 s: acorten el guion")


if __name__ == "__main__":
    main()
