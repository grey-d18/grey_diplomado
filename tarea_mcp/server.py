"""Servidor MCP del BCRP. Empiecen aquí: buscar_series ya funciona; completen los TODO hasta que pasen las pruebas.

    uv run pytest                                           # el contrato: las 8 pruebas deben pasar
    uv run server.py                                        # el servidor, por stdio
    claude mcp add --scope project bcrp -- uv run server.py # conectarlo a Claude Code

Las cuentas las hace Python, no el modelo: así cada cifra sale de una llamada que se puede repetir.
"""
import re
import time
import unicodedata
from functools import cache
from pathlib import Path

import httpx
import pandas as pd
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

AQUI = Path(__file__).parent
CATALOGO = AQUI / "datos" / "catalogo_bcrp.csv"
URL_CATALOGO = "https://estadisticas.bcrp.gob.pe/estadisticas/series/metadata"
API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
SOLO_LECTURA = {"readOnlyHint": True, "openWorldHint": True}

mcp = FastMCP("bcrp", instructions=(
    "Series oficiales del Banco Central de Reserva del Perú (BCRP). Primero busca el código con buscar_series; "
    "luego usa resumen para variaciones, promedios y extremos, y serie para los datos. No calcules a mano: "
    "cita siempre el código de la serie y el periodo."))


def _sin_tildes(s):
    return unicodedata.normalize("NFD", str(s).lower()).encode("ascii", "ignore").decode()


@cache
def catalogo():
    """El catálogo oficial: 17 mil series en unos 7 MB. Se baja una vez y queda en datos/."""
    if not CATALOGO.exists():
        CATALOGO.parent.mkdir(exist_ok=True)
        r = httpx.get(URL_CATALOGO, timeout=120)
        r.raise_for_status()
        CATALOGO.write_text(r.content.decode("latin-1"), encoding="utf-8")
    df = pd.read_csv(CATALOGO, sep=";", dtype=str).fillna("")
    df = df.rename(columns={"Código de serie": "codigo", "Categoría de serie": "categoria", "Grupo de serie": "grupo",
                            "Nombre de serie": "nombre", "Frecuencia": "frecuencia",
                            "Fecha de actualización": "actualizada", "Fecha de inicio": "inicio"})
    df["texto"] = (df.categoria + " " + df.grupo + " " + df.nombre).map(_sin_tildes)
    return df


def _ficha(codigo):
    fila = catalogo().query("codigo == @codigo")
    if fila.empty:
        raise ToolError(f"No existe el código de serie {codigo!r}. Búsquenlo con buscar_series.")
    f = fila.iloc[0]
    return {"codigo": f.codigo, "nombre": f"{f.grupo} - {f.nombre}", "categoria": f.categoria,
            "frecuencia": f.frecuencia, "inicio": f.inicio, "actualizada": f.actualizada}


@mcp.tool(annotations=SOLO_LECTURA)
def buscar_series(texto: str, frecuencia: str | None = None, limite: int = 10) -> list[dict]:
    """Busca series del BCRP por palabras: todas deben aparecer en la categoría, el grupo o el nombre.

    Ejemplos: "ipc 12 meses", "tipo de cambio interbancario venta", "reservas internacionales netas",
    "tasa de referencia", "cobre precio". frecuencia: "Diaria", "Mensual", "Trimestral" o "Anual".
    Devuelve el código, el nombre completo, la frecuencia y desde cuándo hay datos. Si no hay resultados,
    prueben con menos palabras.
    """
    df = catalogo()
    m = pd.Series(True, index=df.index)
    for palabra in _sin_tildes(texto).split():
        m &= df.texto.str.contains(re.escape(palabra))
    if frecuencia:
        m &= df.frecuencia.str.lower() == frecuencia.lower()
    # primero las series vigentes y, entre ellas, las de nombre más corto (las más generales)
    sel = df[m]
    orden = sel.assign(vieja=sel.actualizada < "2025", largo=(sel.grupo + sel.nombre).str.len())
    return [_ficha(c) for c in orden.sort_values(["vieja", "largo"]).head(limite).codigo]


# TODO 1 · serie: la API pide "2025-1" (mensual y trimestral), "2025-01-15" (diaria) o "2025" (anual) y devuelve
# "Ene.2025", "01.Oct.26", "T1.25" o "2025". Aquí se traduce en ambos sentidos.
MESES = {m: i for i, m in enumerate(["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"], 1)}
MESES["sep"] = 9   # setiembre llega como "Set" o como "Sep", según la serie


def _fecha_api(fecha, frecuencia):
    """'2025-01' -> '2025-1' (mensual), '2025-T1' -> '2025-1' (trimestral), '2025-01' -> '2025' (anual)."""
    partes = str(fecha).strip().upper().replace("T", "").split("-")
    frecuencia = frecuencia.lower()
    if frecuencia == "diaria":
        if len(partes) != 3:
            raise ToolError(f"La serie es diaria: pidan la fecha como 2026-10-15, no {fecha!r}.")
        return "-".join(partes)
    if frecuencia == "anual":
        return partes[0]
    if len(partes) < 2:
        raise ToolError(f"La serie es {frecuencia}: pidan la fecha como 2025-01 (o 2025-T1), no {fecha!r}.")
    return f"{partes[0]}-{int(partes[1])}"


def _periodo(nombre):
    """'Ene.2025' -> '2025-01', '01.Oct.26' -> '2026-10-01', 'T1.25' -> '2025-T1', '2025' -> '2025'."""
    p = nombre.split(".")
    if len(p) == 1:
        return p[0]
    if len(p) == 3:
        return f"20{p[2]}-{MESES[p[1].lower()]:02d}-{p[0]}"
    if p[0].upper().startswith("T"):
        return f"20{p[1]}-{p[0].upper()}"
    return f"{p[1]}-{MESES[p[0].lower()]:02d}"


@cache
def _descargar(codigo, desde, hasta):
    """Pide la serie a la API; si responde HTML (límite de pedidos), espera y reintenta. Queda en caché."""
    url = f"{API}/{codigo}/json/{desde}/{hasta}"
    for intento in range(4):
        r = httpx.get(url, timeout=60, follow_redirects=True)
        try:
            return url, r.json().get("periods", [])
        except ValueError:
            time.sleep(2 * (intento + 1))
    raise ToolError(f"El BCRP no respondió con datos para {codigo}. Esperen unos segundos y reintenten: {url}")


@mcp.tool(annotations=SOLO_LECTURA)
def serie(codigo: str, desde: str, hasta: str) -> dict:
    """Datos de una serie del BCRP entre dos fechas, ambas incluidas. Busquen antes el código con buscar_series.

    Fechas según la frecuencia de la serie: "2025-01" (mensual), "2026-10-15" (diaria), "2025-T1" (trimestral),
    "2025" (anual). Ejemplo: serie("PN01273PM", "2026-01", "2026-09"). Devuelve cada periodo con su valor
    y en «fuente» la URL de la API, para citarla. Los días sin dato ("n.d.") se omiten.
    """
    return _serie(codigo, desde, hasta)


def _serie(codigo, desde, hasta):
    f = _ficha(codigo)
    url, periodos = _descargar(codigo, _fecha_api(desde, f["frecuencia"]), _fecha_api(hasta, f["frecuencia"]))
    datos = [{"periodo": _periodo(p["name"]), "valor": float(p["values"][0])}
             for p in periodos if p["values"] and p["values"][0] != "n.d."]
    return {"codigo": codigo, "nombre": f["nombre"], "frecuencia": f["frecuencia"], "datos": datos, "fuente": url}


@mcp.tool(annotations=SOLO_LECTURA)
def resumen(codigo: str, desde: str, hasta: str) -> dict:
    """Estadísticas de una serie del BCRP entre dos fechas: úsenla en vez de calcular a mano.

    Devuelve primero y último (con su periodo), cambio (último - primero), variacion_pct (% de cambio entre ambos),
    promedio, maximo y minimo (con su periodo), n (número de datos) y la fuente. Mismas fechas que serie.
    Ejemplo: resumen("PN01273PM", "2025-12", "2026-09") para ver cómo pasó la inflación de 1.51 % a 4.55 %.
    """
    s = _serie(codigo, desde, hasta)
    if not s["datos"]:
        raise ToolError(f"La serie {codigo} no tiene datos entre {desde} y {hasta}. Revisen las fechas con serie.")
    df = pd.DataFrame(s["datos"])
    primero, ultimo = df.iloc[0], df.iloc[-1]
    punto = lambda fila: {"periodo": fila.periodo, "valor": fila.valor}
    return {"codigo": codigo, "nombre": s["nombre"], "frecuencia": s["frecuencia"], "n": len(df),
            "primero": punto(primero), "ultimo": punto(ultimo),
            "cambio": round(ultimo.valor - primero.valor, 6),
            "variacion_pct": round(100 * (ultimo.valor / primero.valor - 1), 4) if primero.valor else None,
            "promedio": round(df.valor.mean(), 6),
            "maximo": punto(df.loc[df.valor.idxmax()]), "minimo": punto(df.loc[df.valor.idxmin()]),
            "fuente": s["fuente"]}


# TODO 3 · Herramienta propia: el BCRP solo tiene datos del Perú; el Banco Mundial permite compararlo con la región.
WB = "https://api.worldbank.org/v2/country/{paises}/indicator/{indicador}?format=json&date={desde}:{hasta}&per_page=1000"


@mcp.tool(annotations=SOLO_LECTURA)
def comparar_paises(indicador: str = "NY.GDP.MKTP.KD.ZG", paises: str = "PER;CHL;COL;MEX;BRA",
                    desde: int = 2015, hasta: int = 2025) -> dict:
    """Compara al Perú con otros países con datos anuales del Banco Mundial (el BCRP solo publica datos del Perú).

    Úsenla cuando la pregunta compara países o pide el contexto regional, por ejemplo «¿el Perú creció más que
    Chile en 2024?». indicador es un código del Banco Mundial: NY.GDP.MKTP.KD.ZG (crecimiento del PBI, %),
    FP.CPI.TOTL.ZG (inflación, %), SL.UEM.TOTL.ZS (desempleo, %), NE.EXP.GNFS.ZS (exportaciones, % del PBI).
    paises son códigos ISO3 separados por «;». Devuelve los valores por país y año y la URL de la fuente.
    Para cifras oficiales del Perú, prefieran serie o resumen con datos del BCRP.
    """
    url = WB.format(paises=paises.upper(), indicador=indicador, desde=desde, hasta=hasta)
    r = httpx.get(url, timeout=60)
    cuerpo = r.json()
    if len(cuerpo) < 2 or not cuerpo[1]:
        raise ToolError(f"El Banco Mundial no tiene datos de {indicador!r} para {paises!r}: {cuerpo[0]}")
    datos = {}
    for fila in cuerpo[1]:
        if fila["value"] is not None:
            datos.setdefault(fila["country"]["value"], {})[fila["date"]] = round(fila["value"], 4)
    return {"indicador": cuerpo[1][0]["indicator"]["value"], "codigo": indicador,
            "datos": {pais: dict(sorted(anios.items())) for pais, anios in datos.items()}, "fuente": url}


# TODO 4 · Recurso: la ficha de cualquier serie, para adjuntarla con @bcrp:bcrp://serie/PN01273PM
@mcp.resource("bcrp://serie/{codigo}")
def ficha(codigo: str) -> dict:
    """Ficha de una serie del BCRP: nombre, categoría, frecuencia, desde cuándo hay datos y última actualización."""
    return _ficha(codigo)


# Un recurso fijo, además de la plantilla: es el que Claude Code ofrece al escribir @ y adjunta a la conversación.
SERIES_CLAVE = ["PN01273PM", "PN01271PM", "PD04722MM", "PD04638PD", "PN00027MM", "PN38784BM", "PM04923AA",
                "PN38793BM", "PN38063GM"]


@mcp.resource("bcrp://series-clave")
def series_clave() -> list[dict]:
    """Fichas de las series del BCRP más consultadas: inflación, tasa de referencia, tipo de cambio, reservas,
    cobre, oro, PBI y desempleo. Adjúntenla con @ para que Claude tenga los códigos sin tener que buscarlos."""
    return [_ficha(c) for c in SERIES_CLAVE]


# TODO 5 · Prompt: un informe en el que cada cifra se pueda rastrear
@mcp.prompt
def informe(tema: str) -> str:
    """Informe breve sobre un tema económico del Perú, con cifras oficiales del BCRP y verificables."""
    return (f"Escribe un informe de 5 viñetas sobre «{tema}» en el Perú con datos del BCRP. "
            "Busca los códigos con buscar_series y calcula con resumen o serie; no hagas cuentas a mano. "
            "En cada viñeta cita el código de la serie y el periodo, y al final lista las URLs de las fuentes.")


if __name__ == "__main__":
    mcp.run()
