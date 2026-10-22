# %% [markdown]
# # Tarea raster · 5. Exportar para el tablero y contrastar con cifras oficiales
#
# | Paso | Qué hacemos |
# |---|---|
# | 7 · Exportar | Un solo `tablero.json` con todo lo que muestra el tablero (`app.py`) |
# | 8 · Contexto | Cifras oficiales del COEN-INDECI frente a nuestras mediciones, mostradas lado a lado y **sin sumarlas** |

# %%
import json
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

from comun import RAW, PROC, CORTE, CRS_METRICO

# %% [markdown]
# ## Paso 8 · Cifras oficiales
#
# Se buscaron los reportes complementarios del COEN-INDECI de octubre de 2026 para los dos
# distritos (portal.indeci.gob.pe, revisado el 21/10/2026).
#
# | Incendio | Reporte oficial | Inicio | Estado | Área oficial | Daños a personas |
# |---|---|---|---|---|---|
# | Livitaca | COEN-INDECI N.° 14391, 17/10/2026 (Reporte N.° 3), CP Illayuyo | 14/10, 14:30 | Controlado 15/10, extinguido 16/10 | **200 ha** de cobertura natural destruida (EDAN en proceso) | Ninguno reportado |
# | Huanoquite | **No se encontró** reporte en el portal del INDECI | — | — | — | — |
#
# **Validación.** El reporte de Livitaca coincide en lugar y en fechas con los focos VIIRS: el fuego
# empezó el 14/10 a las 14:30, después de la pasada de la tarde, y VIIRS lo detectó en la madrugada
# y la tarde del 15/10. El área oficial (200 ha) es menor que la medida con Sentinel-2 (425–929 ha).
# Esto puede deberse a que la cifra oficial es una evaluación rápida en campo (EDAN) del 15/10 y a
# que en pastizal es difícil estimar a pie la extensión. No se suman ni se promedian: se muestran
# juntas, como en el monitor del profesor.
#
# **Que Huanoquite no tenga reporte** también es un dato: un incendio de 400–900 ha en pastizal, sin
# daños a personas, puede no llegar al registro nacional. Las cifras oficiales subestiman la quema en
# la puna, igual que el umbral 0,27.

# %%
OFICIAL = [
    {"incendio": "Livitaca", "fuente": "COEN-INDECI, Reporte complementario N.° 14391, 17/10/2026, 16:30 h (Reporte N.° 3)",
     "url": "https://portal.indeci.gob.pe/wp-content/uploads/2026/10/REPORTE-COMPLEMENTARIO-N.%C2%BA-14391-17OCT2026-INCENDIO-FORESTAL-EN-EL-DISTRITO-DE-LIVITACA-CUSCO-3.pdf",
     "lugar": "Centro poblado Illayuyo", "inicio": "14/10/2026 14:30", "estado": "Controlado 15/10 16:00; extinguido 16/10 12:30",
     "ha_oficial": 200, "detalle_ha": "cobertura natural destruida (EDAN en proceso)",
     "personas": "No se reportan daños a la vida y la salud"},
    {"incendio": "Huanoquite", "fuente": "No se encontró reporte en portal.indeci.gob.pe (búsqueda del 21/10/2026)",
     "url": None, "lugar": None, "inicio": None, "estado": None, "ha_oficial": None, "detalle_ha": None, "personas": None},
]
pd.DataFrame(OFICIAL)[["incendio", "ha_oficial", "estado", "personas"]]

# %% [markdown]
# ## Paso 7 · Exportar
#
# Las geometrías se simplifican (tolerancia 20 m, el tamaño del píxel) para que el JSON pese poco.

# %%
focos = gpd.read_file(PROC / "focos_limpios.gpkg")
incendios = gpd.read_file(PROC / "incendios_100ha_zona5km.gpkg")
limpieza = pd.read_csv(PROC / "limpieza_focos.csv", index_col=0)["focos_quitados"]
por_distrito = pd.read_csv(PROC / "distritos_personas_5km.csv", dtype={"ubigeo": str})
dnbr = json.loads((PROC / "dnbr_resumen.json").read_text())
cobertura = pd.read_csv(PROC / "que_se_quemo_cusco.csv", header=[0, 1], index_col=[0, 1])
zonas = pd.read_csv(PROC / "personas_viviendas_cusco.csv")
quemado = gpd.read_file(PROC / "area_quemada_cusco.gpkg")

with __import__("rasterio").open(RAW / "worldpop_per_2020_1km.tif") as src:
    nodata = src.nodata
union = zonal_stats([incendios.union_all()], RAW / "worldpop_per_2020_1km.tif", stats=["sum"], nodata=nodata)[0]["sum"]

centros = incendios.to_crs(CRS_METRICO).centroid.to_crs(4326)
lista = incendios.assign(lon=centros.x.round(4), lat=centros.y.round(4))[
    ["distrito", "provincia", "departamento", "ubigeo", "focos", "huella_ha", "frp_mw", "personas_5km", "inicio", "ultimo", "lon", "lat"]]
lista = lista.sort_values("huella_ha", ascending=False).round(1)

# %%
def geojson(geom):
    return json.loads(gpd.GeoSeries([geom], crs=4326).to_crs(CRS_METRICO).simplify(20).to_crs(4326).to_json())

edif_cerca = {}
for inc in quemado.incendio.unique():
    amplia = quemado[(quemado.incendio == inc) & (quemado.cifra == "amplia")].to_crs(CRS_METRICO).geometry.iloc[0]
    caja = gpd.GeoSeries([amplia.buffer(1000)], crs=CRS_METRICO).to_crs(4326).total_bounds
    partes = []
    for t in pd.read_csv(RAW / "open_buildings_v3_points_917.csv.gz", chunksize=2_000_000,
                         usecols=["latitude", "longitude", "confidence"]):
        partes.append(t[t.longitude.between(caja[0], caja[2]) & t.latitude.between(caja[1], caja[3]) & (t.confidence >= 0.75)])
    e = pd.concat(partes)
    e = gpd.GeoDataFrame(e, geometry=gpd.points_from_xy(e.longitude, e.latitude), crs=4326).to_crs(CRS_METRICO)
    e["dist_m"] = e.distance(amplia).round(0)
    e = e[e.dist_m <= 1000]
    edif_cerca[inc] = e[["longitude", "latitude", "dist_m"]].round(5).values.tolist()
    print(f"{inc}: {len(e)} edificaciones a ≤ 1 km")

# %%
ID_INCENDIO = {"Livitaca": 166, "Huanoquite": 12}                   # los mismos del notebook 3
casos = {}
for inc, d in dnbr.items():
    f = focos[focos.incendio == ID_INCENDIO[inc]]
    casos[inc] = {
        **d,
        "cobertura_ha": {c: cobertura.loc[inc]["ha"][c].to_dict() for c in cobertura["ha"].columns},
        "zonas": zonas[zonas.incendio == inc].drop(columns="incendio").round(1).to_dict(orient="records"),
        "area_quemada": {cifra: geojson(quemado[(quemado.incendio == inc) & (quemado.cifra == cifra)].geometry.iloc[0])
                         for cifra in ("conservadora", "amplia")},
        "focos": f[["longitude", "latitude", "t"]].values.tolist(),
        "edificaciones_1km": edif_cerca[inc],
        "oficial": next(o for o in OFICIAL if o["incendio"] == inc),
    }

DECISIONES = [
    ("Datos", "Focos de los 3 satélites VIIRS (S-NPP, NOAA-20, NOAA-21) del 14 al 21 de octubre", "Más pasadas al día = menos fuegos sin detectar"),
    ("Limpieza", "Quitar focos de confianza baja, a < 5 km de volcanes y fuentes fijas (mismo píxel ≥ 5 de 8 días)", "Un foco de calor no es un incendio; una chimenea no se mueve"),
    ("Incendio", "Focos a ≤ 1 km y ≤ 48 h entre sí = un incendio; se analizan los de huella > 100 ha", "Regla del monitor del profesor"),
    ("Ranking nacional", "Personas a 5 km de los focos de cada incendio (WorldPop)", "Convención de 'población cercana' del monitor; comparable entre los 107 incendios"),
    ("Casos", "Livitaca y Huanoquite (Cusco) en vez de Río Tambo o Pucallpa", "En la selva las nubes tapan 33–99 % después del fuego"),
    ("Nubes", "Imagen 'antes' de Huanoquite como compuesto de 10, 7 y 5 de octubre", "Las nubes se mueven; el paisaje antes del fuego casi no cambia"),
    ("Área quemada", "dNBR sembrado con focos: manchas continuas de dNBR ≥ umbral que tocan a ≤ 1 km un foco", "Incluye lo que VIIRS no vio y excluye cambios lejanos"),
    ("Umbral", "Rango: 0,27 (conservador, USGS) a 0,10 (amplio)", "El pasto seco cae poco en NBR aunque se queme del todo"),
    ("Zonas", "Dentro y ≤ 100 m = exposición directa · ≤ 500 m = riesgo directo · ≤ 1 km = medio de vida", "Distancias elegidas por nosotros según lo que cada una permite leer; no 5 km, que se mide desde otro dato"),
    ("Edificaciones", "Open Buildings con confianza ≥ 0,75; se cuentan techos, no hogares", "En la puna, rocas y corrales pueden parecer techos"),
]

tablero = {
    "corte": CORTE,
    "nacional": {
        "focos_limpios": int(len(focos)), "limpieza": limpieza.astype(int).to_dict(),
        "incendios_100ha": int(len(incendios)), "distritos": int(por_distrito.ubigeo.nunique()),
        "personas_5km": round(float(union)),
    },
    "incendios": lista.to_dict(orient="records"),
    "distritos": por_distrito.round(0).to_dict(orient="records"),
    "casos": casos,
    "decisiones": [dict(zip(("tema", "decision", "por_que"), d)) for d in DECISIONES],
}
ruta = PROC / "tablero.json"
ruta.write_text(json.dumps(tablero, ensure_ascii=False, default=str))
print(f"{ruta.name}: {ruta.stat().st_size / 1e6:.2f} MB · personas a 5 km (unión): {union:,.0f}")
