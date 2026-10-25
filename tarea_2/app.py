"""Dashboard Streamlit – Tarea 2: Voces y documentos peruanos, del archivo escaneado al texto buscable.

Solo lee los CSV de datos/, calculados antes en 01_extraccion.ipynb y 02_evaluacion.ipynb:
no carga PaddleOCR, Whisper ni el LLM. Todos los textos ya están seudonimizados, y solo se muestran
las imágenes y los audios de datos/ejemplos/ (licencia libre y sin datos personales, ver preparar_ejemplos.py).

Ejecutar desde la carpeta tarea_2/:
    streamlit run app.py
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
DATOS = BASE / "datos"
TIPOS = {"impreso": "Impreso", "foto": "Foto", "manuscrito": "Manuscrito"}
# colección de cada clip según la descripción de su fuente en Commons (para filtrar los audios por fuente)
COLECCIONES = {
    "Perú-Australia": "Partido Perú-Australia (2022)",
    "Fiestas Patrias": "Fiestas Patrias (2022)",
    "Plaza San Martín": "Protesta en la Plaza San Martín (2022)",
    "Marcha del Orgullo": "Marcha del Orgullo (2022–2023)",
    "fiesta": "Fiestas patronales andinas en Lima (2024)",
    "Semillas de Conocimiento": "Proyecto Semillas de Conocimiento",
    "Salas, pantallas y archivos": "Proyecto Salas, pantallas y archivos",
}
CAMPOS = {
    "tema": "Tema", "resumen": "Resumen", "tipo_documento": "Tipo de documento", "emisor": "Emisor",
    "fecha": "Fecha", "lugar": "Lugar", "monto": "Monto", "numero_hablantes": "Número de hablantes",
    "citas_clave": "Citas clave",
}
COLORES_HABLANTE = ["blue", "green", "orange", "violet", "red", "gray"]

st.set_page_config(page_title="Voces y documentos peruanos", page_icon="📜", layout="wide")


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar_datos() -> dict[str, pd.DataFrame]:
    """Lee una sola vez los CSV publicados (textos seudonimizados y resultados de la evaluación)."""
    archivos = {
        "documentos": "documentos.csv",
        "transcripciones": "transcripciones.csv",
        "extraccion": "extraccion_llm.csv",
        "auditoria": "auditoria_llm.csv",
        "comparativa": "comparativa_modelos.csv",
        "metricas": "metricas_por_caso.csv",
        "fuentes": "fuentes.csv",
        "publicables": "ejemplos/publicables.csv",
    }
    return {nombre: pd.read_csv(DATOS / archivo) for nombre, archivo in archivos.items()}


def sin_tildes(texto: str) -> str:
    """Minúsculas y sin tildes, para que «peru» encuentre «Perú» (la ñ se conserva)."""
    texto = unicodedata.normalize("NFD", str(texto).lower()).replace("ñ", "ñ")
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def coleccion(descripcion: str) -> str:
    return next((nombre for clave, nombre in COLECCIONES.items() if clave in descripcion), "Otra")


def fragmento(texto: str, busqueda: str, ancho: int = 60) -> str:
    """Pedazo del texto alrededor de la primera coincidencia, para ver el contexto en la tabla."""
    plano = " ".join(str(texto).split())
    if not busqueda:
        return plano[:2 * ancho] + "…"
    # sin_tildes conserva la longitud (salvo letras raras), así que la posición sirve en el texto original
    i = sin_tildes(plano).find(sin_tildes(busqueda))
    inicio = max(i - ancho, 0)
    return ("…" if inicio else "") + plano[inicio:i + len(busqueda) + ancho] + "…"


def contiene(textos: pd.Series, busqueda: str) -> pd.Series:
    if not busqueda:
        return pd.Series(True, index=textos.index)
    return textos.fillna("").map(sin_tildes).str.contains(re.escape(sin_tildes(busqueda)))


def a_csv(tabla: pd.DataFrame) -> bytes:
    return tabla.to_csv(index=False).encode("utf-8-sig")  # con BOM, para que Excel lea bien las tildes


def con_hablantes(texto: str) -> str:
    """Convierte [HABLANTE_n] en etiquetas de color y escapa lo que Markdown interpretaría."""
    lineas = []
    for linea in str(texto).splitlines():
        linea = re.sub(r"([*_#`~|<>$])", r"\\\1", linea)
        linea = re.sub(
            r"\[HABLANTE\\_(\d+)\]",
            lambda m: f":{COLORES_HABLANTE[(int(m[1]) - 1) % len(COLORES_HABLANTE)]}[**Hablante {m[1]}:**]",
            linea,
        )
        lineas.append(linea)
    return "  \n".join(lineas)


def ficha_fuente(id_: str) -> None:
    """Metadatos y atribución del archivo original (enlace a su página en Wikimedia Commons)."""
    f = fuentes.loc[id_]
    st.markdown(
        f"**Descripción:** {f['descripcion']}  \n"
        f"**Autor:** {f['autor']} · **Licencia:** {f['licencia']}  \n"
        f"**Fuente:** [Wikimedia Commons]({f['fuente_url']}) · descargado el {f['fecha_descarga']}"
    )


def campos_llm(id_: str) -> None:
    """Campos estructurados que extrajo el LLM y, si el caso se auditó, el veredicto humano por campo."""
    fila = extraccion.loc[extraccion["id"] == id_]
    if fila.empty:
        st.caption("Sin extracción del LLM.")
        return
    campos = json.loads(fila["campos_json"].iloc[0])
    tabla = pd.DataFrame(
        [{"campo": CAMPOS.get(k, k),
          "valor": " · ".join(v) if isinstance(v, list) else ("—" if v in (None, "null") else str(v)),
          "clave": k}
         for k, v in campos.items() if k != "modalidad"]
    )
    revisado = auditoria[auditoria["id"] == id_].set_index("campo")
    if revisado.empty:
        st.caption(f"Extracción de {fila['modelo_llm'].iloc[0]}. Este caso no está en la auditoría humana.")
        st.dataframe(tabla[["campo", "valor"]], hide_index=True, width="stretch")
        return
    tabla["auditoría"] = tabla["clave"].map(revisado["correcto"]).map({1: "✅ correcto", 0: "❌ incorrecto"})
    tabla["comentario"] = tabla["clave"].map(revisado["comentario"])
    st.caption(f"Extracción de {fila['modelo_llm'].iloc[0]}, revisada en la auditoría humana.")
    st.dataframe(tabla[["campo", "valor", "auditoría", "comentario"]].fillna(""), hide_index=True, width="stretch")


def error_por_modelo(id_: str, modelos: list[str], metrica: str) -> None:
    """CER o WER de cada modelo seleccionado en un caso de la muestra de evaluación."""
    caso = metricas[(metricas["id"] == id_) & metricas["modelo"].isin(modelos)]
    if caso.empty:
        st.caption("Ninguno de los modelos seleccionados en la barra lateral.")
        return
    for col, fila in zip(st.columns(len(caso)), caso.itertuples()):
        col.metric(fila.modelo, f"{fila.error:.3f}", help=f"{metrica}; procesado en {fila.segundos:.1f} s")


datos = cargar_datos()
fuentes = datos["fuentes"].set_index("id")
publicables = datos["publicables"].set_index("id")
documentos = datos["documentos"].assign(
    descripcion=lambda d: d["id"].map(fuentes["descripcion"]),
    licencia=lambda d: d["id"].map(fuentes["licencia"]),
)
transcripciones = datos["transcripciones"].assign(
    descripcion=lambda d: d["id"].map(fuentes["descripcion"]),
    licencia=lambda d: d["id"].map(fuentes["licencia"]),
    coleccion=lambda d: d["descripcion"].map(coleccion),
)
extraccion = datos["extraccion"]
auditoria = datos["auditoria"]
comparativa = datos["comparativa"]
metricas = datos["metricas"].query("id != 'total'")  # una fila por caso y modelo

minutos_audio = transcripciones["duracion_seg"].sum() / 60
por_tipo = documentos["tipo"].value_counts()

# ---------------------------------------------------------------- 1. encabezado y contexto
st.markdown(
    "<h1 style='text-align: center;'>¿Pueden convertir documentos escaneados y entrevistas en audio "
    "en un archivo de texto estructurado y buscable?</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Tarea 2 · Voces y documentos peruanos: del archivo escaneado "
    "al texto buscable</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<p style='text-align: center;'>Se extrajo el texto de <b>{len(documentos)} documentos peruanos</b>
({por_tipo.get('impreso', 0)} impresos, {por_tipo.get('foto', 0)} fotos y {por_tipo.get('manuscrito', 0)} manuscritos)
con OCR y de <b>{len(transcripciones)} entrevistas en audio</b> ({minutos_audio:.0f} min) con ASR, se seudonimizó, se
evaluó contra transcripciones manuales y un LLM lo convirtió en campos estructurados.</p>
""",
    unsafe_allow_html=True,
)
with st.expander("Ver la metodología completa"):
    st.markdown(
        """
Todos los archivos vienen de Wikimedia Commons, con licencia libre o en dominio público.

1. **Extracción.** El texto de los documentos se obtuvo con **PaddleOCR** (modelo latino *mobile*) y el de los
   audios con **faster-whisper large-v3-turbo**, con detección de voz (VAD) y separación de hablantes
   (pyannote).
2. **Seudonimización.** Los nombres, DNI y teléfonos de personas reales se reemplazaron por seudónimos
   (`[PERSONA_n]`) antes de publicar cualquier texto.
3. **Evaluación.** Sobre una muestra de 10 documentos y 10 clips con transcripción manual se midió el **CER**
   (documentos) y el **WER** (audio) de tres modelos por modalidad: la línea base, el modelo de la Parte 1 y un
   modelo de otro proyecto (Tesseract y Parakeet).
4. **Extracción estructurada.** Un LLM local (**qwen2.5:3b** con Ollama) extrajo tema, resumen y otros campos de
   cada texto, y se auditaron a mano 20 extracciones.

Todos los resultados se calcularon antes, en los notebooks; esta aplicación solo los muestra.
"""
    )

# ---------------------------------------------------------------- 2. filtros y búsqueda
with st.sidebar:
    st.header("Filtros")
    busqueda = st.text_input("Buscar en los textos OCR y ASR", placeholder="Ej.: decreto, cine, Perú").strip()
    st.caption("No distingue mayúsculas ni tildes.")

    st.subheader("Documentos")
    tipos = st.multiselect("Tipo de documento", list(TIPOS), default=list(TIPOS), format_func=TIPOS.get)

    st.subheader("Modelos evaluados")
    todos_ocr = list(comparativa.query("modalidad == 'documento'")["modelo"])
    todos_asr = list(comparativa.query("modalidad == 'audio'")["modelo"])
    modelos_ocr = st.multiselect("OCR", todos_ocr, default=todos_ocr)
    modelos_asr = st.multiselect("ASR", todos_asr, default=todos_asr)

    st.subheader("Audios")
    dur_min, dur_max = int(transcripciones["duracion_seg"].min()), int(transcripciones["duracion_seg"].max()) + 1
    duracion = st.slider("Duración del clip (s)", dur_min, dur_max, (dur_min, dur_max))
    colecciones = sorted(transcripciones["coleccion"].unique())
    fuentes_audio = st.multiselect("Fuente", colecciones, default=colecciones)

# registros que cumplen los filtros; las tablas y los gráficos de cada sección usan solo estos
docs_f = documentos[documentos["tipo"].isin(tipos) & contiene(documentos["texto_ocr"], busqueda)]
clips_f = transcripciones[
    transcripciones["duracion_seg"].between(*duracion)
    & transcripciones["coleccion"].isin(fuentes_audio)
    & contiene(transcripciones["texto_asr"], busqueda)
]
metricas_f = metricas[
    (metricas["modalidad"].eq("documento") & metricas["modelo"].isin(modelos_ocr) & metricas["id"].isin(docs_f["id"]))
    | (metricas["modalidad"].eq("audio") & metricas["modelo"].isin(modelos_asr) & metricas["id"].isin(clips_f["id"]))
]

# navegación: botones grandes, visibles apenas se abre la página
RESULTADOS, DOCUMENTOS, AUDIOS = "📊 Resultados", "📜 Explorar documentos", "🎙️ Explorar audios"
_, centro, _ = st.columns([1, 3, 1])
with centro:
    st.markdown("<p style='text-align: center; margin-bottom: 0;'><b>Elige qué quieres ver:</b></p>",
                unsafe_allow_html=True)
    seccion = st.segmented_control("Sección", [RESULTADOS, DOCUMENTOS, AUDIOS], default=RESULTADOS,
                                   key="seccion", label_visibility="collapsed", width="stretch")
seccion = seccion or RESULTADOS  # si se vuelve a pulsar el botón activo, queda sin selección
st.divider()

# ---------------------------------------------------------------- 3. indicadores y 4. gráficos
if seccion == RESULTADOS:
    st.subheader("Indicadores clave")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documentos procesados", len(documentos), help="Con OCR: 8 impresos, 8 fotos y 9 manuscritos")
    c2.metric("Minutos de audio transcritos", f"{minutos_audio:.1f}", help=f"{len(transcripciones)} clips con ASR")
    c3.metric("Concordancia LLM vs. auditoría", f"{auditoria['correcto'].mean():.0%}",
              help=f"Campos que la auditoría humana marcó como correctos ({auditoria['correcto'].sum()} de "
                   f"{len(auditoria)}) en 20 extracciones")
    completas = auditoria.groupby("id")["correcto"].min()
    c4.metric("Extracciones completamente correctas", f"{completas.mean():.0%}",
              help=f"{completas.sum()} de {len(completas)} extracciones auditadas sin ningún campo incorrecto")

    for modalidad, metrica, seleccion, muestra in [
        ("documento", "CER", modelos_ocr, "10 documentos"), ("audio", "WER", modelos_asr, "10 clips")
    ]:
        st.markdown(f"**{metrica} total en la muestra de {muestra}** (más bajo es mejor; la diferencia es frente a "
                    "la línea base)")
        filas = comparativa[(comparativa["modalidad"] == modalidad) & comparativa["modelo"].isin(seleccion)]
        base = comparativa.query("modalidad == @modalidad and rol == 'línea base'")["error"].iloc[0]
        if filas.empty:
            st.caption("Ningún modelo seleccionado en la barra lateral.")
            continue
        for col, fila in zip(st.columns(len(filas)), filas.itertuples()):
            extra = f" · RTF {fila.rtf:.2f}" if pd.notna(fila.rtf) else ""
            col.metric(
                fila.modelo, f"{fila.error:.3f}",
                delta=None if fila.rol == "línea base" else f"{fila.error - base:+.3f}", delta_color="inverse",
                help=f"{fila.rol} · {fila.seg_por_caso:.1f} s por caso en {fila.equipo}{extra}",
            )

    st.subheader("Errores por modalidad")
    st.caption("Los gráficos usan solo los casos y modelos que cumplen los filtros de la barra lateral.")
    g1, g2 = st.columns(2)
    docs_m = metricas_f[metricas_f["modalidad"] == "documento"]
    if docs_m.empty:
        g1.info("Ningún documento de la muestra cumple los filtros.")
    else:
        resumen_tipo = docs_m.groupby(["tipo", "modelo"], as_index=False)["error"].mean()
        fig = px.bar(resumen_tipo, x="tipo", y="error", color="modelo", barmode="group",
                     category_orders={"tipo": list(TIPOS)},
                     labels={"error": "CER promedio por documento", "tipo": "Tipo de documento", "modelo": "Modelo"},
                     title="CER de los documentos por tipo y modelo")
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        g1.plotly_chart(fig, width="stretch")
    clips_m = metricas_f[metricas_f["modalidad"] == "audio"]
    if clips_m.empty:
        g2.info("Ningún clip de la muestra cumple los filtros.")
    else:
        fig = px.bar(clips_m, x="id", y="error", color="modelo", barmode="group",
                     labels={"error": "WER", "id": "Clip", "modelo": "Modelo"},
                     title="WER de cada clip de la muestra por modelo")
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        g2.plotly_chart(fig, width="stretch")

    st.subheader("Concordancia del LLM con la auditoría humana")
    acierto = auditoria.assign(campo=auditoria["campo"].map(CAMPOS)) \
        .groupby(["modalidad", "campo"], as_index=False)["correcto"].mean()
    fig = px.bar(acierto, x="correcto", y="campo", color="modalidad", orientation="h", barmode="group",
                 range_x=[0, 1], text_auto=".0%",
                 labels={"correcto": "Proporción de campos correctos", "campo": "", "modalidad": "Modalidad"},
                 title="Acierto del LLM por campo (20 extracciones auditadas)")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Tabla de CER y WER por caso"):
        st.dataframe(
            metricas_f[["modalidad", "id", "tipo", "modelo", "metrica", "error", "segundos"]],
            hide_index=True, width="stretch",
            column_config={"error": st.column_config.NumberColumn("error", format="%.3f")},
        )
        st.download_button("Descargar CSV", a_csv(metricas_f), "metricas_filtradas.csv", "text/csv")

# ---------------------------------------------------------------- 5. exploradores (núcleo de la demo)
def posicion_elegida(clave: str, filtrados: pd.Series, todos: pd.Series) -> int:
    """Posición, entre las opciones, de la fila marcada en la tabla filtrada de abajo; 0 si no hay ninguna."""
    filas = st.session_state.get(clave, {}).get("selection", {}).get("rows", [])
    if not filas or filas[0] >= len(filtrados) or filtrados.iloc[filas[0]] not in list(todos):
        return 0
    return list(todos).index(filtrados.iloc[filas[0]])


def etiquetas(en_muestra: bool, auditado: bool, publicable: bool, medio: str) -> None:
    """Qué se puede juzgar en este registro: referencia y métricas, auditoría del LLM e imagen o audio."""
    st.markdown(" ".join([
        ":green-badge[En la muestra de evaluación]" if en_muestra else ":gray-badge[Sin referencia manual]",
        ":blue-badge[Auditado a mano]" if auditado else ":gray-badge[Sin auditoría]",
        f":orange-badge[{medio} publicable]" if publicable else f":gray-badge[{medio} no publicable]",
    ]))


if seccion == DOCUMENTOS:
    st.subheader("Explorador de documentos")
    st.caption("Elige un documento y compara lo que leyó el OCR con la imagen original y, si existe, con la "
               "transcripción manual. Debajo están los campos que el LLM extrajo de ese texto.")
    c_tipo, c_doc = st.columns([1, 3])
    tipo_exp = c_tipo.radio("Tipo", ["Todos", *TIPOS.values()], horizontal=True, key="tipo_explorador")
    opciones = documentos if tipo_exp == "Todos" else documentos[documentos["tipo"].map(TIPOS) == tipo_exp]
    id_doc = c_doc.selectbox(
        f"Documento ({len(opciones)} de {len(documentos)} procesados)",
        opciones["id"], index=posicion_elegida("tabla_docs", docs_f["id"], opciones["id"]),
        format_func=lambda i: f"{i} · {TIPOS[documentos.set_index('id').loc[i, 'tipo']]} · "
                              f"{fuentes.loc[i, 'descripcion'][:80]}",
    )
    doc = documentos.set_index("id").loc[id_doc]
    pub = publicables.loc[id_doc]
    etiquetas(pd.notna(doc["texto_referencia"]), id_doc in set(auditoria["id"]), pub["publicable"], "Imagen")

    izq, der = st.columns(2)
    with izq:
        if pub["publicable"]:
            st.image(str(BASE / pub["archivo_publico"]),
                     caption=f"{fuentes.loc[id_doc, 'autor']} · {fuentes.loc[id_doc, 'licencia']} "
                             "(Wikimedia Commons), reducida a 1200 px")
        else:
            st.info(f"La imagen no se muestra: {pub['motivo_no_publicable']}. "
                    "El documento se consulta solo como texto seudonimizado y metadatos.")
        ficha_fuente(id_doc)
    with der:
        vistas = ["Texto OCR"] + (["Referencia manual"] if pd.notna(doc["texto_referencia"]) else [])
        for pestana, vista in zip(st.tabs(vistas), vistas):
            with pestana:
                texto = doc["texto_ocr"] if vista == "Texto OCR" else doc["texto_referencia"]
                st.code(texto, language=None, wrap_lines=True, height=420)
        st.caption(f"{len(str(doc['texto_ocr']))} caracteres leídos por {doc['modelo_ocr'].split(' · ')[0]} "
                   "(PaddleOCR mobile latino). Texto seudonimizado.")
        if pd.notna(doc["texto_referencia"]):
            st.markdown("**CER de este documento con cada modelo** (0 = idéntico a la referencia; más bajo es mejor)")
            error_por_modelo(id_doc, modelos_ocr, "CER")
        else:
            st.caption("Este documento no está en la muestra de evaluación: no tiene referencia manual ni CER.")
    st.markdown("**Campos extraídos por el LLM**")
    campos_llm(id_doc)

    st.divider()
    st.subheader("Documentos que cumplen los filtros")
    if docs_f.empty:
        st.warning("Ningún documento cumple los filtros. Cambia la búsqueda o el tipo de documento.")
    else:
        g1, g2 = st.columns([1, 2])
        fig = px.histogram(docs_f.assign(tipo=docs_f["tipo"].map(TIPOS)), x="tipo", color="licencia",
                           category_orders={"tipo": list(TIPOS.values())},
                           labels={"tipo": "Tipo de documento", "licencia": "Licencia"},
                           title="Documentos por tipo y licencia")
        fig.update_layout(yaxis_title="Documentos")
        g1.plotly_chart(fig, width="stretch")
        tabla_docs = docs_f.assign(
            tipo=docs_f["tipo"].map(TIPOS),
            caracteres=docs_f["texto_ocr"].fillna("").str.len(),
            fragmento=docs_f["texto_ocr"].map(lambda t: fragmento(t, busqueda)),
        )
        g2.caption(f"{len(docs_f)} de {len(documentos)} documentos. Haz clic en una fila para abrirla en el "
                   "explorador de arriba.")
        g2.dataframe(tabla_docs[["id", "tipo", "descripcion", "licencia", "caracteres", "fragmento"]],
                     hide_index=True, width="stretch", height=380,
                     on_select="rerun", selection_mode="single-row", key="tabla_docs")
        g2.download_button(
            "Descargar CSV (textos OCR y referencias seudonimizados)",
            a_csv(tabla_docs[["id", "tipo", "descripcion", "licencia", "texto_ocr", "modelo_ocr", "texto_referencia"]]),
            "documentos_filtrados.csv", "text/csv",
        )

if seccion == AUDIOS:
    st.subheader("Explorador de audios")
    st.caption("Elige un clip, escúchalo si es publicable y compara la transcripción automática con la manual. "
               "En los clips de la muestra se compara el WER de los tres modelos.")
    c_col, c_clip = st.columns([1, 3])
    col_exp = c_col.selectbox("Fuente", ["Todas", *sorted(transcripciones["coleccion"].unique())],
                              key="fuente_explorador")
    opciones = transcripciones if col_exp == "Todas" else transcripciones[transcripciones["coleccion"] == col_exp]
    id_clip = c_clip.selectbox(
        f"Clip ({len(opciones)} de {len(transcripciones)} transcritos)",
        opciones["id"], index=posicion_elegida("tabla_clips", clips_f["id"], opciones["id"]),
        format_func=lambda i: f"{i} · {fuentes.loc[i, 'descripcion'][:80]}",
    )
    clip = transcripciones.set_index("id").loc[id_clip]
    pub = publicables.loc[id_clip]
    etiquetas(pd.notna(clip["texto_referencia"]), id_clip in set(auditoria["id"]), pub["publicable"], "Audio")

    izq, der = st.columns(2)
    with izq:
        if pub["publicable"]:
            st.audio(str(BASE / pub["archivo_publico"]), format="audio/mpeg")
            st.caption("Copia en MP3 del fragmento usado, con la licencia del original.")
        else:
            st.info(f"El audio no se reproduce aquí: {pub['motivo_no_publicable']}. Se publica solo la "
                    "transcripción seudonimizada.")
        st.markdown(f"**Duración:** {clip['duracion_seg']:.0f} s · **Hablantes detectados:** {clip['n_hablantes']}")
        ficha_fuente(id_clip)
        if pd.notna(clip["texto_referencia"]):
            st.markdown("**WER de este clip con cada modelo** (0 = idéntico a la referencia; más bajo es mejor)")
            error_por_modelo(id_clip, modelos_asr, "WER")
        else:
            st.caption("Este clip no está en la muestra de evaluación: no tiene referencia manual ni WER.")
    with der:
        vistas = ["Transcripción ASR"] + (["Referencia manual"] if pd.notna(clip["texto_referencia"]) else [])
        for pestana, vista in zip(st.tabs(vistas), vistas):
            with pestana, st.container(height=420):
                if vista == "Transcripción ASR":
                    st.markdown(con_hablantes(clip["texto_asr"]))
                else:
                    st.markdown(con_hablantes(clip["texto_referencia"]))
        st.caption(f"{clip['modelo_asr'].split(' (')[0]}, con marcas de hablante de la diarización. "
                   "Texto seudonimizado.")
    st.markdown("**Campos extraídos por el LLM**")
    campos_llm(id_clip)

    st.divider()
    st.subheader("Clips que cumplen los filtros")
    if clips_f.empty:
        st.warning("Ningún clip cumple los filtros. Cambia la búsqueda, la duración o la fuente.")
    else:
        g1, g2 = st.columns([1, 2])
        fig = px.bar(clips_f.sort_values("duracion_seg"), x="duracion_seg", y="id", color="coleccion",
                     orientation="h", hover_data={"n_hablantes": True},
                     labels={"duracion_seg": "Duración (s)", "id": "", "coleccion": "Fuente",
                             "n_hablantes": "Hablantes"},
                     title="Duración de cada clip por fuente")
        fig.update_layout(legend=dict(orientation="h", y=-0.2), height=520)
        g1.plotly_chart(fig, width="stretch")
        tabla_clips = clips_f.assign(fragmento=clips_f["texto_asr"].map(lambda t: fragmento(t, busqueda)))
        g2.caption(f"{len(clips_f)} de {len(transcripciones)} clips. Haz clic en una fila para abrirla en el "
                   "explorador de arriba.")
        g2.dataframe(
            tabla_clips[["id", "coleccion", "duracion_seg", "n_hablantes", "licencia", "fragmento"]],
            hide_index=True, width="stretch", height=430,
            on_select="rerun", selection_mode="single-row", key="tabla_clips",
            column_config={"duracion_seg": st.column_config.NumberColumn("duración (s)", format="%.0f")},
        )
        g2.download_button(
            "Descargar CSV (transcripciones y referencias seudonimizadas)",
            a_csv(tabla_clips[["id", "coleccion", "duracion_seg", "n_hablantes", "licencia", "texto_asr",
                               "modelo_asr", "texto_referencia"]]),
            "transcripciones_filtradas.csv", "text/csv",
        )
