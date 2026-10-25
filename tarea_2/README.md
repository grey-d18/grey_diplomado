# Tarea 2 – Voces y documentos peruanos: del archivo escaneado al texto buscable

Python Intermedio · Diplomado en Data Science (QLAB)

**Dashboard en vivo:** *(enlace de Streamlit Community Cloud, pendiente de despliegue)*

## 1. Pregunta del proyecto

**¿Pueden convertir documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?**

Se reunieron 25 documentos peruanos (impresos, fotos y manuscritos) y 21 clips de entrevistas en español peruano
(35 min). Su texto se extrajo con OCR y ASR en una laptop de 8 GB, sin GPU dedicada. Luego se seudonimizó, se
evaluó contra transcripciones manuales y se estructuró con un LLM local.

| Modalidad | Modelo | Rol | Error en la muestra | Tiempo |
|---|---|---|---:|---:|
| Documentos (CER, 10 docs) | PaddleOCR por defecto (PP-OCRv6 medium + corrección de deformación) | línea base | 0,319 | 223 s por documento |
| | **PaddleOCR mobile latino** (PP-OCRv5) | Parte 1, **elegido** | **0,269** | 6 s por documento |
| | Tesseract 5 (`spa` / `spa_old`) | otro proyecto | 0,395 | 1 s por documento |
| Audio (WER, 10 clips, 11 min) | faster-whisper small | línea base | 0,219 | RTF 0,26 (CPU) |
| | **faster-whisper large-v3-turbo** | Parte 1, **elegido** | **0,103** | RTF 0,59 (CPU) |
| | NVIDIA Parakeet TDT 0.6B v3 | otro proyecto | 0,351 | RTF 0,41 (GPU, MLX) |

**Extracción estructurada:** qwen2.5:3b (Ollama, temperatura 0) extrajo tema, resumen, tipo de documento, emisor,
fecha, lugar, monto y número de hablantes de los 46 textos. En la auditoría humana de 20 extracciones, el
**82 %** de los campos y el **55 %** de las extracciones (11 de 20) resultaron completamente correctos. Detalle en
`datos/auditoria_llm.csv`.

## 2. Corpus, fuentes y licencias

Todos los archivos vienen de **Wikimedia Commons**. Se descargaron los días **23 a 25 de octubre de 2026**: el
corpus inicial el 23, `doc_028` el 24 y `doc_029` el 25, este último como reemplazo de un documento descartado.
La columna `fecha_descarga` de `datos/fuentes.csv` guarda la fecha de la última regeneración del corpus. El
detalle por archivo (autor, licencia, enlace `?curid=`, procedimiento para recuperarlo) está en
`datos/fuentes.csv` y `datos/atribucion.csv`.

| Corpus | Contenido | Licencias |
|---|---|---|
| 8 impresos | Decretos, normas legales, comunicados del Minsa y *El Peruano* de 1844 | Dominio público (8) |
| 8 fotos | Láminas de la *Nueva corónica* de Guamán Poma, manuscritos y documentos en vitrinas, carteles y paneles | CC BY-SA 2.0 (3), CC BY-SA 4.0 (3), CC BY 2.0 (2) |
| 9 manuscritos | Relación de la Provincia de Saña (s. XVIII), *Relación de antigüedades deste reyno del Pirú* (1613), una partida de bautismo (1792) y el Acta de la Independencia (1821) | Dominio público (7), CC BY-SA 4.0 (1), CC BY-SA 3.0 (1) |
| 21 clips | Entrevistas callejeras (partido Perú-Australia, Fiestas Patrias, Marcha del Orgullo), una arenga en la Plaza San Martín, fiestas patronales andinas y testimonios de los proyectos *Semillas de Conocimiento* y *Salas, pantallas y archivos* | CC BY-SA 4.0 (14), CC BY 4.0 (7) |

Se descartaron y reemplazaron 4 documentos ilegibles (`doc_012`, `doc_013`, `doc_023` y `doc_024`) y un audio
sin voz suficiente (`clip_014`). Los motivos están en `datos/fuentes.csv` y en `01_extraccion.ipynb`.

**Privacidad.** Los nombres de personas reales, DNI, teléfonos y correos se reemplazaron por seudónimos
(`[PERSONA_n]`, `[NOMBRE]`, `[DNI]`, …) con la función `seudonimizar()` de `voz.py` (sesión 13), más una revisión
humana. El diccionario de seudónimos y los archivos originales están en `datos/raw/`, que no se sube a Git. El
dashboard solo muestra las imágenes y los audios de `datos/ejemplos/`: copias reducidas de los archivos con
licencia libre cuyo texto no tuvo ningún dato personal que seudonimizar (20 imágenes y 9 audios). Los demás
registros se consultan solo como texto seudonimizado y metadatos. El criterio está en `preparar_ejemplos.py` y el
resultado por archivo en `datos/ejemplos/publicables.csv`.

## 3. Hallazgos

1. **El tipo de documento pesa más que el modelo.** Los tres OCR leen los impresos casi sin error (CER ≤ 0,07).
   En fotos y manuscritos, en cambio, todos fallan (CER de 0,42 a 0,71 según el modelo), porque ninguno tiene
   reconocimiento de letra manuscrita y el brillo del vidrio rompe la detección de líneas. El peor caso es el Acta
   de la Independencia (CER de 0,69 a 0,87). Ese error se propaga al LLM: con OCR ilegible (`doc_009`, `doc_010`,
   `doc_018`), el LLM inventa el tema, el resumen y el tipo de documento.
2. **En audio, el modelo más grande sigue siendo práctico y el de otro proyecto no generaliza al español peruano.**
   Whisper large-v3-turbo reduce el WER a menos de la mitad del de Whisper small (0,103 frente a 0,219) y sigue
   transcribiendo más rápido que el audio en CPU (RTF 0,59). Parakeet v3, entrenado en 25 idiomas europeos,
   detecta el idioma por su cuenta y cambia al inglés a mitad de algunos clips (español andino en `clip_013`, WER
   0,92), por eso su WER total es 0,351 aunque su mediana sea 0,21. En audio, lo que más sube el error son las
   **varias voces** a la vez y el acento, más que el ruido de la calle.

## 4. Limitaciones técnicas

1. **Referencias sesgadas y muestra pequeña.** Las transcripciones manuales se hicieron corrigiendo la salida de
   los modelos de la Parte 1 (PaddleOCR mobile y Whisper turbo), lo que los favorece. Se nota en `doc_026`, donde
   la Parte 1 queda con CER 0,05 y los otros dos modelos con 0,68. Con 10 casos por modalidad, solo las diferencias
   grandes son concluyentes.
2. **Hardware limitado y latencias no comparables.** Todo corrió en una MacBook de 8 GB de RAM. La línea base de
   PaddleOCR tardó 37 min en 10 documentos (13 min solo en `doc_015`). Parakeet corre en la GPU (MLX) y Whisper en
   la CPU, así que sus RTF no se miden en el mismo equipo. La separación de hablantes (pyannote) sobresegmenta las
   grabaciones de multitud: detecta 9 hablantes en `clip_012` y un hablante extra que solo dice «Gracias.» en
   `clip_020`.

## 5. Ejecutar en local

```bash
cd tarea_2
python3.12 -m venv .venv-app && source .venv-app/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

O, sin crear un entorno, con [uv](https://docs.astral.sh/uv/):

```bash
cd tarea_2
uv run --no-project --with-requirements requirements.txt streamlit run app.py
```

La app se abre en http://localhost:8501. Solo lee los CSV de `datos/`; no necesita PaddleOCR, Whisper ni Ollama.

**Despliegue en Streamlit Community Cloud** (share.streamlit.io): repositorio `grey-d18/grey_diplomado`, rama
`main`, archivo principal `tarea_2/app.py` y Python 3.12 en *Advanced settings*. Streamlit Cloud toma el
`requirements.txt` de la misma carpeta que `app.py`.

### Reproducir los notebooks

Los notebooks usan el entorno del curso `Data-Science-Python` (Python 3.12):

```bash
uv sync --group docai --group audio          # PaddleOCR, faster-whisper, pyannote, jiwer, rapidfuzz
uv pip install plotly kaleido parakeet-mlx   # gráficos del notebook 02 y Parakeet (Apple Silicon)
brew install tesseract tesseract-lang ffmpeg
ollama pull qwen2.5:3b
```

1. Recuperar los archivos crudos en `datos/raw/` con el procedimiento de `datos/fuentes.csv` (columna `acceso`).
2. `01_extraccion.ipynb`: OCR, ASR con VAD y diarización, seudonimización, limpieza y EDA.
3. `02_evaluacion.ipynb`: muestra de evaluación, línea base, extracción con LLM y auditoría, comparación de modelos,
   casos difíciles y decisión.
4. `python preparar_ejemplos.py`: genera la muestra publicable de `datos/ejemplos/`.

## 6. Estructura

```
tarea_2/
├── 01_extraccion.ipynb        # OCR, ASR, seudonimización, limpieza y EDA
├── 02_evaluacion.ipynb        # CER/WER, LLM y auditoría, comparativa de modelos y decisión
├── app.py                     # dashboard Streamlit (solo lee CSV)
├── preparar_ejemplos.py       # muestra ligera publicable para el dashboard
├── requirements.txt           # dependencias del dashboard
├── datos/
│   ├── documentos.csv         # texto OCR seudonimizado (+ referencia en la muestra)
│   ├── transcripciones.csv    # texto ASR con [HABLANTE_n] (+ referencia en la muestra)
│   ├── extraccion_llm.csv     # campos JSON extraídos por el LLM
│   ├── auditoria_llm.csv      # auditoría humana de 20 extracciones
│   ├── comparativa_modelos.csv, metricas_por_caso.csv
│   ├── fuentes.csv, atribucion.csv
│   ├── ejemplos/              # imágenes y audios publicables + publicables.csv
│   └── raw/                   # originales y diccionario de seudónimos (no se sube a Git)
├── modelos/                   # sin pesos en Git
└── evidencias_ia/             # capturas de la conversación con Claude Code
```
