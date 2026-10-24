"""Paso 2 del flujo: del texto largo (PDF) al guion del reel, escena por escena.

Uso:
    python3 guion.py --texto informe.pdf --solo-prompt > prompt.txt
        Imprime el prompt. Péguenlo en chat.qwen.ai, chat.deepseek.com o Claude
        y guarden lo que responda en respuesta.txt.

    python3 guion.py --texto informe.pdf --respuesta respuesta.txt
        Lee la respuesta del chat, la valida y escribe salidas/guion.json.
        Marca con ¡¡ toda cifra que no aparece tal cual en el texto.
"""
import argparse
import json
import os
import re
import sys

try:
    import pymupdf
except ModuleNotFoundError:
    # Si se ejecuta con el python3 del sistema, se relanza con el de .venv
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python, *sys.argv])
    sys.exit("Falta pymupdf: uv pip install --python .venv/bin/python pymupdf")

PALABRAS_MAX = 140      # ~2.4 palabras/s en español ≈ 58 s
DURACION_MAX = 60       # segundos
MAX_CHARS_FUENTE = 40_000
MARCA = "¡¡"

PROMPT = f"""Eres guionista de reels informativos en español para público general.
Escribe el guion de un reel de máximo {DURACION_MAX} segundos ({PALABRAS_MAX} palabras en total) a partir de la fuente de abajo.

Reglas:
- Solo usas datos que aparecen en la fuente. Nada inventado, nada de conocimiento externo, nada exagerado.
- Entre 5 y 8 escenas. La primera es un gancho; la última, un cierre claro.
- "frase" es lo que se dice en voz alta: frases cortas, números escritos en palabras, sin siglas sin explicar.
- "cifra" es el número que se ve en pantalla, copiado TAL CUAL aparece en la fuente (por ejemplo "200 ha" o "92–100 %").
- "etiqueta" explica la cifra en pantalla, en 6 palabras o menos.
- Sin emojis, hashtags ni indicaciones de cámara.

Responde SOLO con un JSON con esta forma, sin texto antes ni después:
{{"titulo": "...", "escenas": [{{"frase": "...", "cifra": "...", "etiqueta": "..."}}]}}

Ejemplo de una escena:
{{"frase": "Cincuenta y seis incendios activos superan las cien hectáreas.", "cifra": "56", "etiqueta": "Incendios activos > 100 ha"}}

Fuente:
\"\"\"
{{fuente}}
\"\"\""""


def leer_fuente(ruta):
    if ruta.lower().endswith(".pdf"):
        with pymupdf.open(ruta) as doc:
            texto = "\n".join(p.get_text() for p in doc)
    else:
        texto = open(ruta, encoding="utf-8").read()
    texto = " ".join(texto.replace("​", "").split())
    if len(texto) > MAX_CHARS_FUENTE:
        print(f"Aviso: fuente recortada a {MAX_CHARS_FUENTE:,} caracteres", file=sys.stderr)
        texto = texto[:MAX_CHARS_FUENTE]
    return texto


def normalizar(s):
    """Minúsculas, espacios simples y sin espacios entre dígitos ("1 824" == "1824")."""
    s = " ".join(s.lower().split())
    return re.sub(r"(?<=\d) (?=\d)", "", s)


def leer_respuesta(ruta):
    """Extrae el JSON de lo que respondió el chat, aunque venga entre ```json ... ```."""
    raw = open(ruta, encoding="utf-8").read()
    inicio, fin = raw.find("{"), raw.rfind("}")
    if inicio < 0 or fin < 0:
        sys.exit(f"No se encontró un JSON en {ruta}")
    return json.loads(raw[inicio:fin + 1])


def validar(guion, fuente):
    fuente_norm = normalizar(fuente)
    avisos = []
    for i, e in enumerate(guion["escenas"], 1):
        faltan = {"frase", "cifra", "etiqueta"} - set(e)
        if faltan:
            sys.exit(f"Escena {i}: faltan los campos {sorted(faltan)}")
        cifra = e["cifra"].removeprefix(MARCA).strip()
        patron = r"(?<![\d,.])" + re.escape(normalizar(cifra)) + r"(?!\d)"
        if cifra and not re.search(patron, fuente_norm):
            e["cifra"] = MARCA + cifra
            avisos.append(f"Escena {i}: la cifra «{cifra}» no aparece tal cual en el texto")
        else:
            e["cifra"] = cifra
        if re.search(r"\d", e["frase"]):
            avisos.append(f"Escena {i}: la frase tiene dígitos; deben ir en palabras")
    palabras = sum(len(e["frase"].split()) for e in guion["escenas"])
    if palabras > PALABRAS_MAX:
        avisos.append(f"El guion tiene {palabras} palabras (máximo {PALABRAS_MAX})")
    return palabras, avisos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texto", required=True, help="informe en PDF o TXT")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--solo-prompt", action="store_true", help="imprime el prompt para pegarlo en un chat")
    modo.add_argument("--respuesta", help="archivo con lo que respondió el chat")
    ap.add_argument("--salida", default="salidas/guion.json")
    args = ap.parse_args()

    fuente = leer_fuente(args.texto)

    if args.solo_prompt:
        print(PROMPT.replace("{fuente}", fuente))
        return

    guion = leer_respuesta(args.respuesta)
    palabras, avisos = validar(guion, fuente)
    guion["revisado"] = False

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(guion, f, ensure_ascii=False, indent=2)

    print(f"{args.salida}: {len(guion['escenas'])} escenas, {palabras} palabras ≈ {palabras / 2.4:.0f} s")
    for a in avisos:
        print(f"  {MARCA} {a}")
    print('Revisen el guion contra la fuente y cambien "revisado" a true para seguir.')


if __name__ == "__main__":
    main()
