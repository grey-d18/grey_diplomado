"""Tablero: ¿a quién alcanzan los incendios de octubre de 2026? Lee datos/procesados/tablero.json (notebook 5)."""
import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

RAIZ = Path(__file__).parent
DATOS = json.loads((RAIZ / "datos" / "procesados" / "tablero.json").read_text())
FIG = RAIZ / "figuras"

st.set_page_config(page_title="Incendios Perú 2026 · ¿a quién alcanza el fuego?", page_icon="🔥", layout="wide")

COLOR = {"conservadora": [122, 46, 31, 200], "amplia": [217, 165, 127, 140], "focos": [31, 78, 156, 230]}
ZONA_COLOR = {"exposición directa": [200, 30, 30], "riesgo directo": [230, 130, 30], "medio de vida": [31, 78, 156]}


def mapa(capas, lon, lat, zoom):
    return pdk.Deck(layers=capas, map_style=pdk.map_styles.CARTO_LIGHT,
                    initial_view_state=pdk.ViewState(longitude=lon, latitude=lat, zoom=zoom),
                    tooltip={"text": "{tooltip}"})


def fmt(x):
    return f"{x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------- encabezado
st.title("¿A quién alcanza el fuego?")
st.markdown(
    "Incendios en el Perú del **14 al 21 de octubre de 2026**, vistos desde el espacio. "
    "Panorama nacional con focos VIIRS y, a fondo, dos incendios en territorio de **comunidades "
    "campesinas quechuas del Cusco**: **Livitaca** (Chumbivilcas) y **Huanoquite** (Paruro).  \n"
    f"Corte de datos: {DATOS['corte']} · Ciencia de Datos con Python · Diplomado QLAB · Reto D")

nac = DATOS["nacional"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Incendios con huella > 100 ha", nac["incendios_100ha"])
c2.metric("Distritos con esos incendios", nac["distritos"])
c3.metric("Personas a 5 km de alguno", fmt(nac["personas_5km"]))
c4.metric("Focos VIIRS tras la limpieza", fmt(nac["focos_limpios"]))

t_nac, t_quemo, t_quien, t_oficial, t_metodo = st.tabs(
    ["Panorama nacional", "¿Qué se quemó?", "¿A quién alcanza?", "Cifras oficiales", "Método y decisiones"])

# ---------------------------------------------------------------- panorama nacional
with t_nac:
    inc = pd.DataFrame(DATOS["incendios"])
    inc["tooltip"] = (inc.distrito.str.title() + " (" + inc.departamento.str.title() + ") · "
                      + inc.huella_ha.map(fmt) + " ha de huella · " + inc.personas_5km.map(fmt) + " personas a 5 km")
    tope = inc.personas_5km.quantile(0.95)
    inc["color"] = inc.personas_5km.clip(upper=tope).map(lambda p: [230, int(200 - 170 * p / tope), 40, 190])
    capa = pdk.Layer("ScatterplotLayer", inc, get_position=["lon", "lat"], get_fill_color="color",
                     get_radius="huella_ha", radius_scale=12, radius_min_pixels=3, pickable=True)
    izq, der = st.columns([3, 2])
    with izq:
        st.pydeck_chart(mapa([capa], -75, -10, 4.3), height=560)
        st.caption("Tamaño del círculo = huella VIIRS (ha). Color = personas a 5 km (amarillo: pocas · rojo: muchas).")
    with der:
        st.markdown("**El fuego más grande no es el que toca a más gente**")
        cols = {"distrito": "Distrito", "departamento": "Depto.", "huella_ha": "Huella (ha)", "personas_5km": "Personas a 5 km"}
        st.markdown("Los más grandes")
        st.dataframe(inc.nlargest(8, "huella_ha")[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
        st.markdown("Los que tienen más gente cerca")
        st.dataframe(inc.nlargest(8, "personas_5km")[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
    st.info("**Cómo leer los 5 km.** En el panorama nacional se cuentan las personas (WorldPop) a 5 km de los "
            "**focos VIIRS** de cada incendio. Es la convención de \"población cercana\" del monitor del profesor y "
            "sirve para comparar los 107 incendios entre sí. No mide daño: un foco de calor no es un incendio, y "
            "WorldPop es un modelo, no un censo.")

# ---------------------------------------------------------------- qué se quemó
casos = DATOS["casos"]
with t_quemo:
    nombre = st.radio("Incendio", list(casos), horizontal=True, key="quemo")
    c = casos[nombre]
    a, b, d, e = st.columns(4)
    a.metric("Área quemada (Sentinel-2)", f"{fmt(c['ha_conservadora'])}–{fmt(c['ha_amplia'])} ha",
             help="dNBR ≥ 0,27 (conservador) a dNBR ≥ 0,10 (amplio), sembrado con focos VIIRS")
    pas = c["cobertura_ha"].get("conservadora (≥ 0,27)", {})
    pas_a = c["cobertura_ha"].get("amplia (≥ 0,10)", {})
    b.metric("Pastizal", f"{fmt(pas.get('Pastizal', 0))}–{fmt(pas_a.get('Pastizal', 0))} ha")
    d.metric("Bosque", f"{fmt(pas.get('Bosque', 0))}–{fmt(pas_a.get('Bosque', 0))} ha")
    e.metric("Cifra oficial (INDECI)", f"{fmt(c['oficial']['ha_oficial'])} ha" if c["oficial"]["ha_oficial"] else "sin reporte")

    izq, der = st.columns([3, 2])
    with izq:
        capas = []
        for cifra in ("amplia", "conservadora"):
            gj = c["area_quemada"][cifra]
            for f in gj["features"]:
                f["properties"]["tooltip"] = f"Área quemada {cifra}"
            capas.append(pdk.Layer("GeoJsonLayer", gj, get_fill_color=COLOR[cifra], stroked=False, pickable=True))
        focos = pd.DataFrame(c["focos"], columns=["lon", "lat", "t"]).assign(tooltip=lambda x: "Foco VIIRS " + x.t + " UTC")
        capas.append(pdk.Layer("ScatterplotLayer", focos, get_position=["lon", "lat"], get_fill_color=COLOR["focos"],
                               get_radius=60, pickable=True))
        st.pydeck_chart(mapa(capas, focos.lon.mean(), focos.lat.mean(), 12), height=480)
        st.caption("Marrón oscuro: quemado con dNBR ≥ 0,27 · marrón claro: solo con dNBR ≥ 0,10 · puntos azules: focos VIIRS.")
    with der:
        cob = pd.DataFrame(c["cobertura_ha"]).fillna(0)
        cob = cob[cob.max(axis=1) >= 1]
        st.markdown("**Qué se quemó** (ha, ESA WorldCover 2021)")
        st.bar_chart(cob.T, horizontal=True, height=220)
        st.markdown(
            "**Por qué un rango.** El dNBR mide cuánto verde se perdió. El bosque parte alto y cae mucho; el pasto "
            "seco de la puna ya tiene poco verde en octubre y, aunque se queme por completo, cae poco. El umbral "
            "0,27 (USGS, pensado para bosques) deja fuera parte de esa quema; 0,10 la incluye, con más riesgo de "
            "contar pasto que solo se secó.")
        imagenes = pd.DataFrame({"fecha": list(c["nubes_pct"]), "nubes en la zona (%)": list(c["nubes_pct"].values())})
        imagenes["uso"] = ["antes"] * (len(imagenes) - 1) + ["después"]
        st.markdown("**Imágenes Sentinel-2 usadas**")
        st.dataframe(imagenes, hide_index=True, width="stretch")
    with st.expander("¿Cuánto cambia el número según la decisión?"):
        st.dataframe(pd.Series(c["sensibilidad_ha"], name="ha").to_frame(), width="stretch")
        st.image(str(FIG / "03_dnbr_livitaca_huanoquite.png"), caption="Falso color SWIR y dNBR de los dos incendios")

# ---------------------------------------------------------------- a quién alcanza
with t_quien:
    nombre = st.radio("Incendio", list(casos), horizontal=True, key="quien")
    c = casos[nombre]
    st.warning(
        "**Las distancias son una decisión nuestra**, medidas desde el borde del área quemada (amplia):  \n"
        "• **Dentro y hasta 100 m → exposición directa**: el fuego pasó por ahí o llegó \"a la puerta\".  \n"
        "• **Hasta 500 m → riesgo directo**: poblados que el fuego pudo alcanzar si seguía avanzando.  \n"
        "• **Hasta 1 km → medio de vida**: familias cuyos pastos de uso diario se quemaron.  \n"
        "Aquí **no** usamos 5 km: ese radio se mide desde los focos VIIRS y solo sirve para el panorama nacional. "
        "**Exposición no es daño**: el satélite no dice si una casa se quemó.")
    z = pd.DataFrame(c["zonas"]).set_index("zona")
    significado = {"dentro del área quemada": "exposición directa", "hasta 100 m": "exposición directa",
                   "hasta 500 m": "riesgo directo", "hasta 1 km": "medio de vida"}
    cols = st.columns(4)
    for col, (zona, fila) in zip(cols, z.iterrows()):
        col.metric(f"Edificaciones {zona}", fmt(fila["edificaciones (conf ≥ 0,75)"]), help=significado[zona])
    izq, der = st.columns([3, 2])
    with izq:
        e = pd.DataFrame(c["edificaciones_1km"], columns=["lon", "lat", "dist_m"])
        e["tipo"] = pd.cut(e.dist_m, [-1, 100, 500, 1000], labels=["exposición directa", "riesgo directo", "medio de vida"])
        e["color"] = e.tipo.astype(str).map(ZONA_COLOR)
        e["tooltip"] = "Edificación a " + e.dist_m.astype(int).astype(str) + " m del área quemada · " + e.tipo.astype(str)
        gj = c["area_quemada"]["amplia"]
        for f in gj["features"]:
            f["properties"]["tooltip"] = "Área quemada (amplia)"
        capas = [pdk.Layer("GeoJsonLayer", gj, get_fill_color=COLOR["amplia"], stroked=False, pickable=True),
                 pdk.Layer("ScatterplotLayer", e, get_position=["lon", "lat"], get_fill_color="color",
                           get_radius=25, radius_min_pixels=2, pickable=True)]
        st.pydeck_chart(mapa(capas, e.lon.mean(), e.lat.mean(), 12), height=480)
        st.caption("Rojo: exposición directa (≤ 100 m) · naranja: riesgo directo (≤ 500 m) · azul: medio de vida (≤ 1 km). "
                   "Edificaciones de Google Open Buildings con confianza ≥ 0,75.")
    with der:
        tabla = z[["edificaciones (conf ≥ 0,75)", "personas (WorldPop)"]].copy()
        tabla.insert(0, "significa", [significado[i] for i in tabla.index])
        st.dataframe(tabla.round(0), width="stretch")
        st.markdown(
            "**Lectura.** En **Livitaca** las casas están **dispersas**: estancias familiares junto a sus pastos, y por "
            "eso más casas quedan pegadas al fuego. En **Huanoquite** están **agrupadas**: el fuego se detuvo a pocos "
            "cientos de metros de dos poblados.  \n\n"
            "**WorldPop a esta escala.** Pone gente dentro del área quemada, donde casi no hay techos: reparte la "
            "población en toda la celda de 1 km. Para un incendio, las edificaciones dicen más.")

# ---------------------------------------------------------------- cifras oficiales
with t_oficial:
    st.markdown("Cifras oficiales y mediciones satelitales **lado a lado, sin sumarlas**.")
    filas = []
    for nombre, c in casos.items():
        o = c["oficial"]
        filas.append({"incendio": nombre, "fuente oficial": o["fuente"], "inicio": o["inicio"], "estado": o["estado"],
                      "área oficial (ha)": o["ha_oficial"], "daños a personas": o["personas"],
                      "Sentinel-2 (ha)": f"{fmt(c['ha_conservadora'])}–{fmt(c['ha_amplia'])}",
                      "huella VIIRS (ha)": round(c["huella_viirs_ha"])})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.markdown(
        "- **Livitaca:** el reporte del COEN-INDECI coincide en lugar y fechas con los focos VIIRS. Su cifra (200 ha) "
        "es una evaluación rápida en campo y es menor que la satelital.  \n"
        "- **Huanoquite:** no se encontró reporte. Un incendio de 400–900 ha en pastizal, sin daños a personas, puede "
        "no llegar al registro nacional: las cifras oficiales también invisibilizan la quema en la puna.")
    url = casos["Livitaca"]["oficial"]["url"]
    st.markdown(f"Fuente: [COEN-INDECI, Reporte complementario N.° 14391]({url})")

# ---------------------------------------------------------------- método
with t_metodo:
    st.markdown("Los ocho pasos del monitor del profesor, y cada decisión que tomamos.")
    st.dataframe(pd.DataFrame(DATOS["decisiones"]).rename(columns={"tema": "Tema", "decision": "Decisión", "por_que": "Por qué"}),
                 hide_index=True, width="stretch")
    st.markdown(
        "**Limitaciones**  \n"
        "- Un foco de calor no es un incendio; incluye quemas agrícolas.  \n"
        "- Las nubes esconden el fuego: la selva (Río Tambo, Pucallpa) queda fuera del dNBR.  \n"
        "- WorldPop es un modelo, no un censo; WorldCover es de 2021.  \n"
        "- Open Buildings cuenta techos (viviendas, cocinas, depósitos), no hogares.  \n"
        "- Exposición no es daño.\n\n"
        "Datos: NASA FIRMS (VIIRS), Sentinel-2 L2A y ESA WorldCover vía Microsoft Planetary Computer, WorldPop 2020, "
        "Google Open Buildings v3, límites INEI (peru-geojson), COEN-INDECI.")
