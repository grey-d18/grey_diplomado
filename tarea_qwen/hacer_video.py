"""Paso 5 del flujo: del audio narrado al reel con subtítulos (Whisper + Pillow + ffmpeg).

Uso:
    python3 hacer_video.py

Lee salidas/guion.json, salidas/escenas.json y salidas/reel.wav (de voz.py).
Escribe en salidas/:
    reel.mp4     1080×1920, 30 fps, H.264 + AAC, −14 LUFS
    reel.srt     subtítulos aparte (LinkedIn, Facebook)
    portada.png  portada del reel
"""
import argparse
import difflib
import json
import os
import subprocess
import sys

try:
    import mlx_whisper
except ModuleNotFoundError:
    # Si se ejecuta con el python3 del sistema, se relanza con el de .venv
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python, *sys.argv])
    sys.exit("Falta el entorno: uv pip install --python .venv/bin/python mlx-whisper pillow num2words")

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from voz import MODELO_ASR, normalizar

ANCHO, ALTO, FPS = 1080, 1920, 30
MARGEN_X = 90                     # zona segura: los bordes los tapan los botones de cada app
ARRIBA, ABAJO = 250, 1500
CIERRE = 2.5                      # segundos de tarjeta final

FONDO = (27, 23, 20)
CREMA = (246, 241, 234)
TENUE = (178, 168, 156)
NARANJA = (240, 122, 58)

AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
DIN = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"


def fuente(tam, estilo="Bold"):
    indices = {"Bold": 0, "Demi": 2, "Medium": 5, "Regular": 7, "Heavy": 8}
    return ImageFont.truetype(AVENIR, tam, index=indices[estilo])


def envolver(draw, texto, f, ancho):
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if draw.textlength(prueba, font=f) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    return lineas + [actual] if actual else lineas


def centrado(draw, y, texto, f, color, espacio=1.15):
    for linea in envolver(draw, texto, f, ANCHO - 2 * MARGEN_X):
        draw.text((ANCHO / 2, y), linea, font=f, fill=color, anchor="mt")
        y += int(f.size * espacio)
    return y


def cifra_ajustada(draw, texto, tam=240):
    while tam > 80 and draw.textlength(texto, font=ImageFont.truetype(DIN, tam)) > ANCHO - 2 * MARGEN_X:
        tam -= 10
    return ImageFont.truetype(DIN, tam)


def pastilla_ia(draw, y):
    f = fuente(32, "Demi")
    texto = "Voz generada con IA"
    w = draw.textlength(texto, font=f)
    x0 = (ANCHO - w) / 2 - 26
    draw.rounded_rectangle((x0, y, x0 + w + 52, y + 58), radius=29, outline=TENUE, width=2)
    draw.text((ANCHO / 2, y + 29), texto, font=f, fill=TENUE, anchor="mm")


def capa_base(guion):
    img = Image.new("RGB", (ANCHO, ALTO), FONDO)
    d = ImageDraw.Draw(img)
    d.text((ANCHO / 2, ARRIBA), "INCENDIOS · CUSCO · OCTUBRE 2026", font=fuente(32, "Demi"), fill=NARANJA, anchor="mt")
    centrado(d, ARRIBA + 60, guion["titulo"], fuente(54), CREMA)
    pastilla_ia(d, ABAJO - 110)
    return img


def capa_escena(escena):
    img = Image.new("RGBA", (ANCHO, ALTO), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = cifra_ajustada(d, escena["cifra"])
    d.text((ANCHO / 2, 640), escena["cifra"], font=f, fill=CREMA, anchor="mt")
    centrado(d, 640 + f.size + 10, escena["etiqueta"], fuente(52, "Medium"), TENUE)
    return img


def tiempos_de_palabras(guion, escenas, reel_wav):
    """Toma las palabras del guion (bien escritas) y les pone los tiempos que oyó Whisper."""
    palabras = [(p, i) for i, e in enumerate(guion["escenas"]) for p in e["frase"].split()]
    r = mlx_whisper.transcribe(reel_wav, path_or_hf_repo=MODELO_ASR, language="es", word_timestamps=True)
    oidas = []  # una entrada por token normalizado: "1824" se vuelve tres palabras
    for s in r["segments"]:
        for w in s["words"]:
            tokens = normalizar(w["word"]).split()
            paso = (w["end"] - w["start"]) / max(len(tokens), 1)
            oidas += [(tok, w["start"] + k * paso, w["start"] + (k + 1) * paso) for k, tok in enumerate(tokens)]

    guion_norm = [normalizar(p) for p, _ in palabras]
    inicio, fin = [None] * len(palabras), [None] * len(palabras)
    sm = difflib.SequenceMatcher(None, guion_norm, [t for t, _, _ in oidas], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            inicio[a + k], fin[a + k] = oidas[b + k][1], oidas[b + k][2]

    # Las palabras que Whisper oyó distinto (p. ej. "Libitaca") se reparten entre sus vecinas,
    # sin salir de los límites de su escena
    for i, (_, esc) in enumerate(palabras):
        if inicio[i] is None:
            prev = next((fin[j] for j in range(i - 1, -1, -1) if fin[j] is not None and palabras[j][1] == esc),
                        escenas[esc]["inicio"])
            nxt = next((inicio[j] for j in range(i + 1, len(palabras)) if inicio[j] is not None
                        and palabras[j][1] == esc), escenas[esc]["fin"])
            huecos = [j for j in range(i, len(palabras)) if inicio[j] is None and palabras[j][1] == esc]
            huecos = huecos[:next((k for k, j in enumerate(huecos) if j != i + k), len(huecos))]
            paso = (nxt - prev) / len(huecos)
            for k, j in enumerate(huecos):
                inicio[j], fin[j] = prev + k * paso, prev + (k + 1) * paso
    for i, (_, esc) in enumerate(palabras):
        inicio[i] = max(inicio[i], escenas[esc]["inicio"])
        fin[i] = max(min(fin[i], escenas[esc]["fin"]), inicio[i])
    return [{"palabra": p, "escena": e, "inicio": a, "fin": b} for (p, e), a, b in zip(palabras, inicio, fin)]


def agrupar(palabras, maximo):
    grupos, actual = [], []
    for w in palabras:
        if actual and (len(actual) == maximo or w["escena"] != actual[-1]["escena"]
                       or w["inicio"] - actual[-1]["fin"] > 0.35):
            grupos.append(actual)
            actual = []
        actual.append(w)
        if w["palabra"][-1] in ".,;:?!":
            grupos.append(actual)
            actual = []
    return grupos + [actual] if actual else grupos


def escribir_srt(grupos, ruta):
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(ruta, "w", encoding="utf-8") as f:
        for n, g in enumerate(grupos, 1):
            texto = " ".join(w["palabra"] for w in g)
            fin = g[-1]["fin"] + 0.1
            if n < len(grupos):
                fin = min(fin, grupos[n][0]["inicio"])
            f.write(f"{n}\n{ts(g[0]['inicio'])} --> {ts(fin)}\n{texto}\n\n")


def dibujar_subtitulo(d, grupo, t):
    f = fuente(66)
    lineas = envolver(d, " ".join(w["palabra"] for w in grupo), f, ANCHO - 2 * MARGEN_X)
    y, k = 1170, 0
    for linea in lineas:
        n = len(linea.split())
        trozo = grupo[k:k + n]
        ancho = d.textlength(linea, font=f)
        x = (ANCHO - ancho) / 2
        for w in trozo:
            suena = w["inicio"] <= t < w["fin"] + 0.05
            d.text((x, y), w["palabra"], font=f, fill=NARANJA if suena else CREMA)
            x += d.textlength(w["palabra"] + " ", font=f)
        y += int(f.size * 1.2)
        k += n


def tarjeta_final(guion):
    img = Image.new("RGB", (ANCHO, ALTO), FONDO)
    d = ImageDraw.Draw(img)
    d.text((ANCHO / 2, ARRIBA), "INCENDIOS · CUSCO · OCTUBRE 2026", font=fuente(32, "Demi"), fill=NARANJA, anchor="mt")
    y = centrado(d, 700, guion["titulo"], fuente(72), CREMA)
    centrado(d, y + 50, "Estudio independiente con datos abiertos: Sentinel-2, NASA FIRMS VIIRS, "
                        "Google Open Buildings y COEN-INDECI.", fuente(40, "Medium"), TENUE, 1.3)
    pastilla_ia(d, ABAJO - 110)
    return img


def portada(guion, base):
    img = base.copy()
    d = ImageDraw.Draw(img)
    principal = guion["escenas"][0]
    f = cifra_ajustada(d, principal["cifra"], 300)
    d.text((ANCHO / 2, 620), principal["cifra"], font=f, fill=NARANJA, anchor="mt")
    centrado(d, 620 + f.size + 20, principal["etiqueta"], fuente(56, "Medium"), CREMA)
    return img


def medir_volumen(wav):
    """Primera pasada de loudnorm: mide el audio para normalizarlo bien en la segunda."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", wav, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    salida = r.stderr
    return json.loads(salida[salida.rindex("{"):salida.rindex("}") + 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="salidas")
    args = ap.parse_args()
    s = args.salida

    guion = json.load(open(f"{s}/guion.json", encoding="utf-8"))
    escenas = json.load(open(f"{s}/escenas.json", encoding="utf-8"))
    reel_wav = f"{s}/reel.wav"
    dur_audio = sf.info(reel_wav).duration
    total = dur_audio + CIERRE

    palabras = tiempos_de_palabras(guion, escenas, reel_wav)
    escribir_srt(agrupar(palabras, 7), f"{s}/reel.srt")
    grupos = agrupar(palabras, 4)

    base = capa_base(guion)
    portada(guion, base).save(f"{s}/portada.png")
    capas = [capa_escena(e) for e in guion["escenas"]]
    final = tarjeta_final(guion)

    m = medir_volumen(reel_wav)
    filtro_audio = (f"apad,loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                    f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
                    "linear=true")
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ANCHO}x{ALTO}", "-r", str(FPS), "-i", "-",
        "-i", reel_wav, "-af", filtro_audio, "-ar", "48000", "-t", f"{total:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", f"{s}/reel.mp4"], stdin=subprocess.PIPE)

    cuadros = int(total * FPS)
    for n in range(cuadros):
        t = n / FPS
        if t >= dur_audio:
            alfa = min(1.0, (t - dur_audio) / 0.3)
            cuadro = Image.blend(base, final, alfa)
        else:
            cuadro = base.copy()
            esc = next((i for i, e in enumerate(escenas) if t < e["fin"] + 0.3), len(escenas) - 1)
            alfa = min(1.0, max(0.0, (t - escenas[esc]["inicio"] + 0.15) / 0.3))
            capa = capas[esc]
            if alfa < 1:
                capa = capa.copy()
                capa.putalpha(capa.getchannel("A").point(lambda v: int(v * alfa)))
            cuadro.paste(capa, (0, 0), capa)
            d = ImageDraw.Draw(cuadro)
            grupo = next((g for g in grupos if g[0]["inicio"] - 0.05 <= t <= g[-1]["fin"] + 0.25), None)
            if grupo:
                dibujar_subtitulo(d, grupo, t)
            # barra de avance
            d.rectangle((MARGEN_X, ABAJO - 20, MARGEN_X + (ANCHO - 2 * MARGEN_X) * t / dur_audio, ABAJO - 14),
                        fill=NARANJA)
        ffmpeg.stdin.write(cuadro.tobytes())
        if n % (FPS * 5) == 0:
            print(f"  {t:4.0f} s / {total:.0f} s", flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"{s}/reel.mp4 · {s}/reel.srt · {s}/portada.png ({total:.1f} s)")


if __name__ == "__main__":
    main()
