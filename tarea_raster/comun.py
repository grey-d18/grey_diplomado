"""Rutas y funciones compartidas por los notebooks del estudio de incendios."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph

RAIZ = Path(__file__).parent
RAW = RAIZ / "datos" / "raw"
PROC = RAIZ / "datos" / "procesados"
FIG = RAIZ / "figuras"
for carpeta in (RAW, PROC, FIG):
    carpeta.mkdir(parents=True, exist_ok=True)

CORTE = "2026-10-21"                       # fecha de descarga de los focos
CRS_METRICO = 32718                        # UTM 18S: metros, cubre la selva central
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Volcanes activos del sur: sus anomalías térmicas no son incendios
VOLCANES = {"Sabancaya": (-71.857, -15.787), "Ubinas": (-70.903, -16.355),
            "Misti": (-71.409, -16.294)}


def leer_distritos() -> gpd.GeoDataFrame:
    d = gpd.read_file(RAW / "peru_distrital_simple.geojson")
    return d[["IDDIST", "NOMBDIST", "NOMBPROV", "NOMBDEP", "geometry"]]


def leer_focos() -> gpd.GeoDataFrame:
    """Une los tres satélites VIIRS (S-NPP, NOAA-20, NOAA-21) y deja solo los focos dentro del Perú."""
    partes = []
    for ruta in sorted(RAW.glob("firms_viirs_*_7d_*.csv")):
        partes.append(pd.read_csv(ruta).assign(sat=ruta.name.split("_")[2]))
    f = pd.concat(partes, ignore_index=True)
    f["t"] = pd.to_datetime(f.acq_date + " " + f.acq_time.astype(str).str.zfill(4),
                            format="%Y-%m-%d %H%M")          # hora UTC
    g = gpd.GeoDataFrame(f, geometry=gpd.points_from_xy(f.longitude, f.latitude), crs=4326)
    return gpd.sjoin(g, leer_distritos(), predicate="within").drop(columns="index_right")


def agrupar_incendios(focos: gpd.GeoDataFrame, km: float = 1.0, horas: float = 48) -> np.ndarray:
    """Un incendio = focos conectados a ≤ km y ≤ horas entre sí (VIIRS mide en píxeles de 375 m; un fuego puede reavivarse al día siguiente)."""
    m = focos.to_crs(CRS_METRICO)
    grafo = radius_neighbors_graph(np.c_[m.geometry.x, m.geometry.y], km * 1000,
                                   mode="connectivity").tocoo()
    t = focos["t"].values.astype("datetime64[h]").astype(np.int64)
    cerca_en_tiempo = np.abs(t[grafo.row] - t[grafo.col]) <= horas
    grafo.data = grafo.data * cerca_en_tiempo
    grafo.eliminate_zeros()
    _, etiquetas = connected_components(grafo.tocsr(), directed=False)
    return etiquetas
