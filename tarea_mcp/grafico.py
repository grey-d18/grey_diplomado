"""Figura del informe: aciertos, «no sé» y errores por modo, a partir de resultados.csv y resultados_sin_null.csv.

    uv run grafico.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).parent
base = pd.read_csv(AQUI / "resultados.csv")
sin_null = pd.read_csv(AQUI / "resultados_sin_null.csv").assign(modo="memoria sin null")
res = pd.concat([base, sin_null])
filas = {"mcp": "Con MCP del BCRP", "memoria": "Solo memoria", "memoria sin null": "Solo memoria,\nsin permiso de decir «no sé»"}
categorias = {"acierta": "#0ca30c", "no sabe": "#b5b3ad", "se equivoca": "#d03b3b"}
tabla = res.groupby("modo").resultado.value_counts().unstack(fill_value=0).reindex(columns=categorias, fill_value=0)
tabla = tabla.loc[list(filas)[::-1]]
segundos = res.groupby("modo").segundos.mean()

fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
izq = pd.Series(0, index=tabla.index)
for cat, color in categorias.items():
    ax.barh([filas[m] for m in tabla.index], tabla[cat], left=izq, color=color, height=0.55,
            edgecolor="#fcfcfb", linewidth=2, label=cat)
    for i, (m, n) in enumerate(tabla[cat].items()):
        if n:
            ax.text(izq[m] + n / 2, i, str(n), ha="center", va="center", fontsize=10,
                    color="#ffffff" if cat != "no sabe" else "#0b0b0b", fontweight="bold")
    izq += tabla[cat]
for i, m in enumerate(tabla.index):
    ax.text(10.2, i, f"{segundos[m]:.1f} s por pregunta", va="center", fontsize=9, color="#52514e")
ax.set_xlim(0, 12.6); ax.set_xticks(range(0, 11, 2))
ax.set_xlabel("Preguntas (de 10)", color="#52514e")
ax.tick_params(colors="#52514e", length=0)
for lado in ["top", "right", "left"]:
    ax.spines[lado].set_visible(False)
ax.spines["bottom"].set_color("#d6d4cd")
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
ax.set_title("¿Acierta Claude Haiku las cifras del BCRP?", loc="left", fontsize=12, color="#0b0b0b", pad=30)
fig.tight_layout()
fig.savefig(AQUI / "grafico.png", facecolor=fig.get_facecolor())
