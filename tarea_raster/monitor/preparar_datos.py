"""Prepara los datos del monitor web (un JSON por fuente) a partir de datos/procesados.
Las imágenes y la malla 3D salen de preparar_imagenes.py."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "datos" / "procesados"
SALIDA = Path(__file__).resolve().parent / "datos"
SALIDA.mkdir(exist_ok=True)
t = json.loads((PROC / "tablero.json").read_text())


def guardar(nombre, filas):
    (SALIDA / f"{nombre}.json").write_text(json.dumps(filas, ensure_ascii=False, separators=(",", ":")))
    print(f"{nombre:14s} {len(filas):6d} filas  {(SALIDA / f'{nombre}.json').stat().st_size / 1e3:8.1f} KB")


# Panorama nacional
n = t["nacional"]
guardar("nacional", [{"incendios_100ha": n["incendios_100ha"], "distritos": n["distritos"], "personas_5km": n["personas_5km"],
                      "focos_limpios": n["focos_limpios"], "quitados_confianza_baja": n["limpieza"]["confianza baja"],
                      "quitados_fuente_fija": n["limpieza"]["fuente fija (≥ 5 días)"], "quitados_volcan": n["limpieza"]["volcán (< 5 km)"],
                      "corte": t["corte"]}])
inc = pd.DataFrame(t["incendios"])
inc["id"] = [f"{r.distrito.title()} #{i + 1}" for i, r in inc.reset_index(drop=True).iterrows()]
inc["inicio"] = inc.inicio.str[:10]; inc["ultimo"] = inc.ultimo.str[:10]
guardar("incendios", inc[["id", "distrito", "provincia", "departamento", "focos", "huella_ha", "personas_5km",
                          "inicio", "ultimo", "lon", "lat"]].assign(distrito=lambda d: d.distrito.str.title(),
                          provincia=lambda d: d.provincia.str.title(), departamento=lambda d: d.departamento.str.title()).to_dict("records"))
dep = gpd.read_file(RAIZ / "datos" / "raw" / "peru_distrital_simple.geojson").dissolve("NOMBDEP").reset_index()
dep["geometry"] = dep.to_crs(32718).simplify(2500).to_crs(4326)
guardar("departamentos", [{"departamento": r.NOMBDEP.title(), "geometria": json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"]}
                          for r in dep.itertuples()])

# Los dos incendios del Cusco
casos, cob, zon, sens, edif, focos, celdas, areas = [], [], [], [], [], [], [], []
for nombre, c in t["casos"].items():
    o = c["oficial"]
    nubes = c["nubes_pct"]
    fechas_antes = c["imagenes_antes"]
    casos.append({"incendio": nombre, "distrito": c["distrito"].title(), "provincia": c["provincia"].title(),
                  "fuego_inicio": c["fuego_viirs"][0], "fuego_fin": c["fuego_viirs"][1], "focos": c["focos"],
                  "huella_viirs_ha": round(c["huella_viirs_ha"]), "ha_conservadora": round(c["ha_conservadora"] - 1e-6),  # empate x,5 hacia abajo, como en la tabla de sensibilidad
                  "ha_amplia": round(c["ha_amplia"]),
                  "imagen_antes": " + ".join(fechas_antes), "nubes_antes_pct": nubes[fechas_antes[0]],
                  "imagen_despues": c["imagen_despues"], "nubes_despues_pct": nubes[c["imagen_despues"]],
                  "ha_oficial": o["ha_oficial"], "fuente_oficial": o["fuente"], "estado_oficial": o["estado"],
                  "personas_oficial": o["personas"], "url_oficial": o["url"]})
    for cifra, valores in c["cobertura_ha"].items():
        for cobertura, ha in valores.items():
            if ha >= 1:
                cob.append({"incendio": nombre, "cifra": "conservadora" if "0,27" in cifra else "amplia",
                            "cobertura": cobertura, "ha": round(ha)})
    significado = {"dentro del área quemada": "exposición directa", "hasta 100 m": "exposición directa",
                   "hasta 500 m": "riesgo directo", "hasta 1 km": "medio de vida"}
    for z in c["zonas"]:
        zon.append({"incendio": nombre, "zona": z["zona"], "significa": significado[z["zona"]],
                    "edificaciones": int(z["edificaciones (conf ≥ 0,75)"]), "personas_worldpop": round(z["personas (WorldPop)"])})
    for variante, ha in c["sensibilidad_ha"].items():
        sens.append({"incendio": nombre, "variante": variante, "ha": round(ha)})
    for lon, lat, d in c["edificaciones_1km"]:
        franja = "exposición directa" if d <= 100 else "riesgo directo" if d <= 500 else "medio de vida"
        edif.append({"incendio": nombre, "lon": lon, "lat": lat, "distancia_m": int(d), "franja": franja})
    for lon, lat, hora in c["focos"]:
        focos.append({"incendio": nombre, "lon": round(lon, 5), "lat": round(lat, 5), "hora_utc": hora})
    for k, cel in enumerate(c["worldpop_celdas"]):
        (x0, y0), _, (x1, y1), _ = cel["poligono"]
        celdas.append({"incendio": nombre, "celda": k, "lon0": round(x0, 5), "lat0": round(y0, 5), "lon1": round(x1, 5),
                       "lat1": round(y1, 5), "personas": cel["personas"], "personas_km2": cel["densidad"]})
    for cifra in ("conservadora", "amplia"):
        geom = c["area_quemada"][cifra]["features"][0]["geometry"]
        areas.append({"incendio": nombre, "cifra": cifra, "ha": round(c[f"ha_{cifra}"]), "geometria": geom})

guardar("casos", casos); guardar("cobertura", cob); guardar("zonas", zon); guardar("sensibilidad", sens)
guardar("edificaciones", edif); guardar("focos_casos", focos); guardar("worldpop", celdas)
guardar("area_quemada", areas); guardar("decisiones", t["decisiones"])
