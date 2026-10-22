"""Imágenes del monitor web: Sentinel-2 en color real (antes y después), dNBR
y una malla de altitud para la vista 3D de cada incendio, más el relieve del Perú para el mapa nacional.

Todo se guarda en monitor/web/ en EPSG:4326, con sus límites en monitor/web/capas.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
from matplotlib.colors import LightSource
from PIL import Image
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays
from shapely import contains_xy

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "datos" / "procesados"
WEB = Path(__file__).resolve().parent / "web"
WEB.mkdir(exist_ok=True)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
catalogo = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
s2 = catalogo.get_collection("sentinel-2-l2a")
resumen = json.loads((PROC / "dnbr_resumen.json").read_text())

# Para el color real usamos la fecha "antes" sin nubes sobre el incendio
ANTES_COLOR = {"Livitaca": "2026-10-07", "Huanoquite": "2026-10-05"}
LADO = 1100            # píxeles del lado mayor de cada imagen (~10 m por píxel)
MALLA = 240            # celdas del lado mayor de la malla 3D (~45 m)
capas = json.loads((WEB / "capas.json").read_text()) if (WEB / "capas.json").exists() else {}


def a_4326(da, ancho, alto, limites, metodo=Resampling.bilinear):
    """Reproyecta a una rejilla lon/lat fija para que todas las capas calcen píxel a píxel."""
    from rasterio.transform import from_bounds
    t = from_bounds(*limites, ancho, alto)
    return da.rio.reproject("EPSG:4326", transform=t, shape=(alto, ancho), resampling=metodo)


def color_real(item, limites, ancho, alto):
    off = -1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
    canales = []
    for b in ("B04", "B03", "B02"):
        da = rioxarray.open_rasterio(item.assets[b].href).squeeze(drop=True)
        caja = gpd.GeoSeries.from_xy([limites[0], limites[2]], [limites[1], limites[3]], crs=4326).to_crs(da.rio.crs).total_bounds
        da = da.rio.clip_box(caja[0] - 400, caja[1] - 400, caja[2] + 400, caja[3] + 400).astype("float32")   # margen: sin bordes vacíos
        da = a_4326(da, ancho, alto, limites)
        canales.append((da.values + off) / 10000)
    rgb = np.clip(np.dstack(canales), 0, None)
    rgb = np.clip(rgb / 0.28, 0, 1) ** (1 / 1.25)        # estiramiento suave: la puna es clara
    return Image.fromarray((rgb * 255).astype("uint8"))


import sys
for nombre, c in (resumen.items() if "--solo-peru" not in sys.argv else []):
    with rasterio.open(PROC / f"dnbr_{nombre.lower()}.tif") as r:
        lim = rasterio.warp.transform_bounds(r.crs, 4326, *r.bounds)
    m = 0.02                                               # recorte del borde: evita franjas sin dato al reproyectar
    lim = (lim[0] + m * (lim[2] - lim[0]), lim[1] + m * (lim[3] - lim[1]), lim[2] - m * (lim[2] - lim[0]), lim[3] - m * (lim[3] - lim[1]))
    dx, dy = lim[2] - lim[0], lim[3] - lim[1]
    k = np.cos(np.radians((lim[1] + lim[3]) / 2))
    ancho, alto = (LADO, round(LADO * dy / (dx * k))) if dx * k >= dy else (round(LADO * dx * k / dy), LADO)
    print(nombre, lim, ancho, alto)

    escenas = {e[11:19]: e for e in c["escenas"]}
    antes = s2.get_item(escenas[ANTES_COLOR[nombre].replace("-", "")])
    despues = s2.get_item(escenas[c["imagen_despues"].replace("-", "")])
    color_real(antes, lim, ancho, alto).save(WEB / f"{nombre.lower()}_antes.jpg", quality=84)
    color_real(despues, lim, ancho, alto).save(WEB / f"{nombre.lower()}_despues.jpg", quality=84)

    # dNBR por clases USGS, transparente donde no hay quema
    d = rioxarray.open_rasterio(PROC / f"dnbr_{nombre.lower()}.tif", masked=True).squeeze(drop=True)
    d = a_4326(d.rio.write_nodata(np.nan, encoded=False), ancho, alto, lim, Resampling.nearest).values
    d[~np.isfinite(d) | (np.abs(d) > 2)] = np.nan
    rgba = np.zeros((alto, ancho, 4), "uint8")
    for lo, col in [(0.10, (247, 205, 120, 150)), (0.27, (236, 122, 44, 205)), (0.44, (198, 42, 28, 225)), (0.66, (110, 20, 40, 235))]:
        rgba[np.nan_to_num(d, nan=-1) >= lo] = col
    Image.fromarray(rgba).save(WEB / f"{nombre.lower()}_dnbr.png", optimize=True)

    # Relieve: Copernicus DEM 30 m → sombreado (capa de luz y sombra) y malla 3D
    items = catalogo.search(collections=["cop-dem-glo-30"], bbox=lim).item_collection()
    dem = merge_arrays([rioxarray.open_rasterio(it.assets["data"].href).squeeze(drop=True).rio.clip_box(lim[0] - .01, lim[1] - .01, lim[2] + .01, lim[3] + .01)
                        for it in items])
    z = a_4326(dem.astype("float32"), ancho, alto, lim).values

    nc, nf = (MALLA, round(MALLA * alto / ancho)) if ancho >= alto else (round(MALLA * ancho / alto), MALLA)
    zm = a_4326(dem.astype("float32"), nc, nf, lim).values
    zm = np.where(np.isfinite(zm), zm, np.nanmin(zm))
    (WEB / f"{nombre.lower()}_malla.json").write_text(json.dumps(
        {"filas": nf, "cols": nc, "z": np.round(zm).astype(int).ravel().tolist()}, separators=(",", ":")))
    capas[nombre] = {"limites": [round(v, 6) for v in lim], "ancho": ancho, "alto": alto,
                     "fecha_antes": ANTES_COLOR[nombre], "fecha_despues": c["imagen_despues"],
                     "z_min": int(np.nanmin(z)), "z_max": int(np.nanmax(z))}

# Relieve nacional: Copernicus DEM 90 m, leído en su resumen de resumen 1:4 (~370 m)
peru = gpd.read_file(RAIZ / "datos" / "raw" / "peru_distrital_simple.geojson").union_all()
LIM_PE = (-81.45, -18.4, -68.6, 0.05)
items = catalogo.search(collections=["cop-dem-glo-90"], bbox=LIM_PE).item_collection()
print("teselas DEM 90 m:", len(items))


def baja(it):
    try:
        return rioxarray.open_rasterio(it.assets["data"].href, overview_level=1).squeeze(drop=True)
    except Exception as e:
        print("sin resumen", it.id, e)
        return None


with ThreadPoolExecutor(12) as ex:
    piezas = [p for p in ex.map(baja, items) if p is not None]
dem = merge_arrays(piezas)
AN = 1000
AL = round(AN * (LIM_PE[3] - LIM_PE[1]) / (LIM_PE[2] - LIM_PE[0]) / np.cos(np.radians(-9)))
z = a_4326(dem.astype("float32"), AN, AL, LIM_PE).values
z = np.where(np.isfinite(z) & (z > -100), z, 0)
paso = (LIM_PE[2] - LIM_PE[0]) * np.cos(np.radians(9)) * 111320 / AN
som = LightSource(315, 40).hillshade(z, vert_exag=6, dx=paso, dy=paso)
xs = np.linspace(LIM_PE[0], LIM_PE[2], AN)
ys = np.linspace(LIM_PE[3], LIM_PE[1], AL)
xx, yy = np.meshgrid(xs, ys)
dentro = contains_xy(peru, xx, yy)
# Capa de luz y sombra: negro translúcido en las laderas en sombra, blanco en las iluminadas,
# así funciona sobre el color de tierra del tema claro y del oscuro
img = np.zeros((AL, AN, 4), "uint8")
luz = som - 0.72
img[..., :3] = np.where(luz[..., None] > 0, 255, 0).astype("uint8")
img[..., 3] = np.where(dentro, np.clip(np.abs(luz) * np.where(luz > 0, 520, 330), 0, 200), 0).astype("uint8")
Image.fromarray(img).save(WEB / "peru_relieve.png", optimize=True)
capas["peru"] = {"limites": list(LIM_PE), "ancho": AN, "alto": AL}
(WEB / "capas.json").write_text(json.dumps(capas, ensure_ascii=False, indent=1))
for f in sorted(WEB.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e3:8.1f} KB")
