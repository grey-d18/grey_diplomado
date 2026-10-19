# Tarea 1 – Noticias peruanas: del titular al clasificador

Python Intermedio · Diplomado en Data Science (QLAB) · **Opción B – El Comercio**

## 1. Pregunta del proyecto

**¿Pueden convertir titulares peruanos en un clasificador de noticias?**

Sí. Se recolectaron 399 noticias de El Comercio, un LLM local las etiquetó en 4 categorías
(`politica`, `economia`, `deportes`, `ciencia_tecnologia`), una persona auditó 20 etiquetas y,
con esas etiquetas, se ajustó DistilBERT multilingüe. En un test reservado de 60 noticias el
modelo alcanza **F1 macro 0.838**, frente a 0.663 de un modelo zero-shot, y clasifica un titular
en ~6–10 ms en CPU. Un dashboard en Streamlit permite explorar el corpus y clasificar titulares nuevos.

| Resultado (test de 60 noticias, referencia: sección del diario) | Accuracy | F1 macro |
|---|---:|---:|
| Zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli` | 0.683 | 0.663 |
| LLM débil Qwen 2.5 3B (Ollama) | 0.867 | 0.863 |
| **DistilBERT fine-tuned** | **0.833** | **0.838** |

Línea base zero-shot en 40 titulares (Paso 1): accuracy 0.700, F1 macro 0.679.
Acuerdo LLM–humano en la auditoría de 20 casos al azar: **18/20 (90 %)**. Los dos desacuerdos
fueron un informe sobre la caída de la fauna silvestre (LLM: ciencia y tecnología; revisión
humana: economía, por su impacto en la economía de la región) y una nota sobre estafas digitales
en la venta de entradas para BTS (LLM: economía; revisión humana: ciencia y tecnología, por
tratarse de ciberseguridad). Detalle en `datos/auditoria_humana.csv`.

## 2. Medio y fechas de extracción

- **Medio:** El Comercio (`elcomercio.pe`).
- **Fecha de descarga:** 18 de octubre de 2026.
- **Fechas de publicación:** del 6 al 18 de octubre de 2026 (la ventana varía por categoría; ver hallazgo 2).
- **Fuente:** páginas de archivo por sección (`/archivo/politica/`, `/archivo/economia/`,
  `/archivo/deporte-total/`, `/archivo/tecnologia/`), permitidas por `robots.txt`.
  No hubo bloqueo; el RSS oficial de cada sección quedó como respaldo automático.
- **Buenas prácticas:** `User-Agent` académico, pausas de 1–2 s, máximo 1 reintento y respaldo
  crudo en `datos/raw/` (no versionado por tamaño).
- **Limpieza:** 400 extraídas → 399 (1 URL duplicada). La tabla de mapeo sección → categoría
  y la revisión manual de los casos dudosos están en `01_recoleccion.ipynb`.

## 3. Hallazgos analíticos

**Hallazgo 1 – Un modelo pequeño entrenado con etiquetas de un LLM casi iguala al LLM, cientos de veces más rápido.**
DistilBERT obtiene F1 0.838 frente a 0.863 del LLM que le dio las etiquetas, pero tarda ~6–10 ms
por titular frente a ~5.8–7.6 s y usa ~0.6 GB de RAM frente a ~2.5 GB. Supera ampliamente al
zero-shot (0.663). Por eso es el modelo elegido para el dashboard; el LLM es más útil como
etiquetador fuera de línea.

**Hallazgo 2 – Política "absorbe" las noticias institucionales.** Es la categoría con peor F1 (0.778)
por baja precisión (0.667): noticias de deportes sobre arbitraje y VAR, o de economía sobre
Indecopi y la SBS, se predicen como política. El LLM ya tenía ese sesgo (envió 16 noticias de
economía a política) y DistilBERT lo heredó. En cambio, cuando el modelo predice *deportes*
acierta siempre (precisión 1.000).

Además, la ventana temporal es desigual: deportes reúne ~100 noticias en 3 días, mientras que
ciencia y tecnología necesita 13 días. El corpus está balanceado en cantidad, no en tiempo.

## 4. Limitaciones técnicas

**Limitación 1 – Supervisión débil y referencia imperfecta.** El modelo se entrena con etiquetas
de un LLM, que coinciden con la sección del diario en el 89.7 % de los casos: sus errores
sistemáticos (economía → política) pasan al clasificador. La referencia de evaluación, la sección
del diario, tampoco es perfecta: algunas secciones son formatos (opinión, ecdata, BBC) y se
asignaron según la página de archivo donde aparecieron. El test tiene solo 60 noticias, así que
un acierto más o menos mueve el F1 en ~1.5 puntos.

**Limitación 2 – Cobertura y recursos.** Un solo medio, menos de dos semanas de noticias y una
sola descarga: el modelo puede no generalizar a otros medios, fechas o temas. Todo se ejecutó en
CPU con 8 GB de RAM, por lo que se eligieron modelos pequeños (Qwen 3B en lugar de 7B, BERT base
en lugar de XLM-RoBERTa large). Las latencias se midieron en ese mismo equipo y varían según su carga (entre ejecuciones el LLM osciló entre ~5.8 y ~7.6 s por noticia). La probabilidad que muestra el dashboard no está calibrada.

## 5. Ejecución en local

Requisitos: Python 3.12 y, solo para volver a etiquetar con el LLM, [Ollama](https://ollama.com).
Los comandos se ejecutan **dentro de la carpeta `tarea_1/`**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Pesos del modelo.** `modelo/model.safetensors` pesa ~540 MB y no se sube a GitHub (límite de
100 MB). Para generarlo, ejecutar `02_clasificacion.ipynb` (el entrenamiento tarda ~1 min en CPU).
Sin los pesos, el dashboard funciona igual pero muestra un aviso en el probador en vivo.

Para reproducir todo el proceso:

```bash
ollama pull qwen2.5:3b              # LLM local (~2 GB)
jupyter lab                         # ejecutar 01_recoleccion.ipynb y luego 02_clasificacion.ipynb
```

Las etiquetas del LLM ya están en `datos/etiquetas_llm.csv`; el notebook las reutiliza y no
vuelve a consultar Ollama salvo para las noticias que falten.

## Estructura

```text
tarea_1/
├── 01_recoleccion.ipynb     # scraping, limpieza y EDA
├── 02_clasificacion.ipynb   # zero-shot, LLM, auditoría, fine-tuning y evaluación
├── app.py                   # dashboard Streamlit
├── requirements.txt
├── datos/
│   ├── noticias.csv         # 399 noticias, 7 columnas
│   ├── etiquetas_llm.csv    # etiquetas de Qwen 2.5 3B
│   ├── auditoria_humana.csv # 20 casos revisados a mano
│   └── particion.csv        # split train / val / test
└── modelo/                  # config, tokenizer y metricas.json (pesos fuera de git)
```
