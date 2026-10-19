"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador.

Ejecutar desde la carpeta tarea_1/:
    streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
DIR_MODELO = BASE / "modelo"
CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {
    "politica": "Política",
    "economia": "Economía",
    "deportes": "Deportes",
    "ciencia_tecnologia": "Ciencia y tecnología",
}

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")


# ---------------------------------------------------------------- datos y modelo
@st.cache_data
def cargar_noticias() -> pd.DataFrame:
    return pd.read_csv(BASE / "datos" / "noticias.csv").fillna({"bajada": ""})


@st.cache_data
def cargar_metricas() -> dict:
    archivo = DIR_MODELO / "metricas.json"
    return json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}


@st.cache_resource
def cargar_clasificador():
    """Carga DistilBERT una sola vez por sesión del servidor, en CPU."""
    if not (DIR_MODELO / "model.safetensors").exists():
        return None
    import torch
    from transformers import pipeline

    torch.set_num_threads(2)  # equipo de 8 GB: dejar núcleos libres
    return pipeline("text-classification", model=str(DIR_MODELO), device="cpu", top_k=None)


noticias = cargar_noticias()
metricas = cargar_metricas()

# ---------------------------------------------------------------- 1. encabezado
st.title("¿Pueden convertir titulares peruanos en un clasificador de noticias?")
st.markdown(
    f"""
**Metodología.** Se recolectaron **{len(noticias)} noticias de El Comercio** (archivo HTML por
sección, respetando `robots.txt`). Un LLM local (**{metricas.get('modelo_llm', 'Qwen 2.5 3B')}** con
Ollama) etiquetó cada noticia en 4 categorías y una persona auditó 20 casos al azar. Con esas
etiquetas se ajustó **DistilBERT multilingüe** (HuggingFace `Trainer`, 3 épocas), evaluado en un
conjunto de test reservado y comparado con un modelo zero-shot.
"""
)

# ---------------------------------------------------------------- 2. filtros
st.sidebar.header("Filtros")
medios = st.sidebar.multiselect("Medio", sorted(noticias.medio.unique()), default=sorted(noticias.medio.unique()))
categorias = st.sidebar.multiselect(
    "Categoría", CATEGORIAS, default=CATEGORIAS, format_func=NOMBRES.get
)
filtradas = noticias[noticias.medio.isin(medios) & noticias.etiqueta.isin(categorias)]
st.sidebar.caption(f"{len(filtradas)} de {len(noticias)} noticias seleccionadas")

# ---------------------------------------------------------------- 3. indicadores
f1 = metricas.get("distilbert_test", {}).get("f1_macro")
acuerdo = metricas.get("acuerdo_llm_humano")
n_audit = metricas.get("auditoria_n", 20)

c1, c2, c3 = st.columns(3)
c1.metric("Noticias recolectadas", f"{len(noticias)}", help="Total del corpus, antes de filtros")
c2.metric("F1 macro en test (DistilBERT)", f"{f1:.3f}" if f1 is not None else "—",
          help=f"Sobre {metricas.get('n_test', '—')} noticias de test reservadas")
c3.metric("Concordancia LLM vs auditoría humana",
          f"{acuerdo:.0%}" if acuerdo is not None else "pendiente",
          help=f"{round(acuerdo * n_audit) if acuerdo is not None else '—'}/{n_audit} casos revisados a mano")

# ---------------------------------------------------------------- 4. visualización y tabla
izq, der = st.columns([2, 3])
with izq:
    conteo = (filtradas.etiqueta.value_counts().reindex(categorias, fill_value=0)
              .rename_axis("etiqueta").reset_index(name="noticias"))
    conteo["categoría"] = conteo.etiqueta.map(NOMBRES)
    fig = px.bar(conteo, x="categoría", y="noticias", color="categoría", text="noticias",
                 title="Distribución de categorías (según filtros)")
    fig.update_layout(showlegend=False, height=420, margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")
with der:
    st.subheader("Noticias")
    st.dataframe(filtradas[["fecha", "etiqueta", "titular", "bajada", "url"]],
                 width="stretch", height=360, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("url")})
    st.download_button("Descargar CSV filtrado", filtradas.to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

# ---------------------------------------------------------------- 5. probador en vivo
st.divider()
st.header("Probador en vivo")
st.write("Escribe o pega un titular nuevo y el modelo predecirá su categoría.")

clasificador = cargar_clasificador()
if clasificador is None:
    st.warning("No se encontró `modelo/model.safetensors`. Ejecuta `02_clasificacion.ipynb` "
               "para entrenar y guardar el modelo (los pesos no se suben a GitHub por tamaño).")
else:
    titular = st.text_input("Titular", placeholder="Ej.: BCR mantiene su tasa de referencia en 4,25%")
    if st.button("Clasificar", type="primary", disabled=not titular.strip()):
        resultado = sorted(clasificador(titular.strip())[0], key=lambda r: -r["score"])
        mejor = resultado[0]
        st.success(f"**{NOMBRES[mejor['label']]}** — probabilidad {mejor['score']:.1%}")
        probs = pd.DataFrame({"categoría": [NOMBRES[r["label"]] for r in resultado],
                              "probabilidad": [r["score"] for r in resultado]})
        fig = px.bar(probs, x="probabilidad", y="categoría", orientation="h", text_auto=".1%",
                     range_x=[0, 1], title="Probabilidad por categoría")
        fig.update_layout(height=280, yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("La probabilidad es la salida softmax del modelo; no está calibrada, así que "
                   "indica confianza relativa, no la probabilidad real de acierto.")
