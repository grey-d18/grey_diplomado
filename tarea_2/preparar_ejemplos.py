"""Prepara datos/ejemplos/: la muestra ligera de imágenes y audios que sí se puede publicar en el dashboard.

Criterio de publicación (los demás registros se consultan solo como texto y metadatos):
  1. Licencia libre o dominio público (todo el corpus viene de Wikimedia Commons).
  2. Sin datos personales: la seudonimización no reemplazó ningún nombre, DNI, teléfono ni correo en su
     texto OCR/ASR ni en su referencia. Una imagen o un audio mostraría tal cual lo que el texto oculta.
  3. Excepciones decididas a mano (EXCLUIDOS), por si algo se ve o se oye y no quedó en el texto.

Las imágenes se reducen a 1200 px de lado mayor (JPEG calidad 75) y los audios se pasan a MP3 mono de 48 kbps,
para que el repositorio y la app sigan siendo ligeros. Necesita Pillow y ffmpeg; se ejecuta en local, una vez:

    python preparar_ejemplos.py
"""
import re
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = Path(__file__).parent
DATOS = BASE / "datos"
SALIDA = DATOS / "ejemplos"
SEUDONIMOS = re.compile(r"\[(?:PERSONA_\d+|NOMBRE|DNI|TELEFONO|CORREO)\]")
LICENCIAS_LIBRES = ("Dominio público", "CC BY", "CC0")
EXCLUIDOS: dict[str, str] = {}  # id -> motivo

fuentes = pd.read_csv(DATOS / "fuentes.csv").set_index("id")
registros = pd.concat([
    pd.read_csv(DATOS / "documentos.csv").rename(columns={"texto_ocr": "texto", "archivo": "original"}),
    pd.read_csv(DATOS / "transcripciones.csv").rename(columns={"texto_asr": "texto", "archivo_audio": "original"}),
])[["id", "original", "texto", "texto_referencia"]]

SALIDA.mkdir(exist_ok=True)
filas = []
for r in registros.itertuples():
    licencia = fuentes.loc[r.id, "licencia"]
    entidades = len(SEUDONIMOS.findall(f"{r.texto} {r.texto_referencia}"))
    if not licencia.startswith(LICENCIAS_LIBRES):
        motivo = f"licencia {licencia}"
    elif entidades:
        motivo = f"datos personales ({entidades} seudónimos en el texto)"
    else:
        motivo = EXCLUIDOS.get(r.id, "")
    publico = ""
    if not motivo:
        origen = BASE / r.original
        if r.id.startswith("doc"):
            publico = f"datos/ejemplos/{r.id}.jpg"
            imagen = Image.open(origen).convert("RGB")
            imagen.thumbnail((1200, 1200))
            imagen.save(BASE / publico, quality=75, optimize=True)
        else:
            publico = f"datos/ejemplos/{r.id}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(origen),
                            "-ac", "1", "-b:a", "48k", str(BASE / publico)], check=True)
    filas.append({"id": r.id, "licencia": licencia, "publicable": not motivo,
                  "archivo_publico": publico, "motivo_no_publicable": motivo})

tabla = pd.DataFrame(filas)
tabla.to_csv(SALIDA / "publicables.csv", index=False)
print(tabla.groupby(tabla["id"].str[:3])["publicable"].agg(["sum", "count"]))
print(f"Tamaño de datos/ejemplos/: {sum(f.stat().st_size for f in SALIDA.iterdir()) / 1e6:.1f} MB")
