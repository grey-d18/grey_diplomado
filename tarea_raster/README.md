# Tarea raster – Incendios en el Perú vistos desde el espacio

Ciencia de Datos con Python · Diplomado en Data Science (QLAB) · 21 de octubre de 2026
**Reto D – ¿A quién afecta?**, con lectura antropológica.

## 1. Pregunta

**¿A quién alcanzan los incendios de octubre de 2026, y qué es lo que se quema en los territorios
de comunidades campesinas quechuas del Cusco?**

- En todo el país: personas a 5 km de cada incendio (WorldPop).
- A fondo, en dos incendios del Cusco con cielo despejado: **Livitaca** (Chumbivilcas) y
  **Huanoquite** (Paruro). Se mide el área quemada (dNBR) y qué cobertura se quemó.

## 2. Datos y fechas

| Dato | Fuente | Fecha | Tipo |
|---|---|---|---|
| Focos de calor (S-NPP, NOAA-20, NOAA-21) | NASA FIRMS, VIIRS «South America 7d» | 14–21 oct 2026, descarga 21 oct | Puntos |
| Límites distritales | `juaneladio/peru-geojson` (INEI, 1 834 distritos) | — | Polígonos |
| Población | WorldPop 2020, 1 km, *unconstrained* | 2020 | Raster |
| Imágenes | Sentinel-2 L2A, Planetary Computer | ver tabla del método | Raster, 20 m |
| Cobertura | ESA WorldCover | 2021 | Raster, 10 m |
| Relieve | Copernicus DEM GLO-30 | — | Raster |
| Edificaciones | Google Open Buildings v3 (bloques S2 `911` y `917`) | 2023 | Puntos |

## 3. Método (los 8 pasos del monitor del profesor)

| Paso | Notebook | Qué se hizo y qué se decidió |
|---|---|---|
| 1 · Descargar | `01_datos_mapa_base.ipynb` | Todo público y sin clave; cada archivo se baja una vez |
| 2 · Mapa base | `01` | Distritos y relieve |
| 3 · Cobertura | `01`, `03` | WorldCover; en el notebook 3 se cruza a 20 m con el área quemada |
| 4 · Focos | `02_focos_incendios.ipynb` | 9 949 → 8 587 focos: −1 336 de confianza baja, −26 fuentes fijas (mismo píxel de 375 m con calor en ≥ 5 de 8 días), 0 cerca de volcanes |
| 5 · Incendios | `02` | Focos a ≤ 1 km **y** ≤ 48 h = un incendio (grafo + componentes conectadas). Huella = píxeles de 375 m × 14,06 ha. Se analizan los de huella > 100 ha |
| 6 · Raster | `03_raster_cusco.ipynb` | NBR con B8A y B12 (20 m), reflectancia corregida (offset −1000), máscara SCL de nubes, sombras y agua. **dNBR sembrado con focos**: manchas continuas con dNBR ≥ umbral que tocan a ≤ 1 km un foco VIIRS. Dos umbrales: 0,27 (conservador) y 0,10 (amplio, para pastizal) |
| 7 · Exportar | pendiente | JSON para el tablero |
| 8 · Contexto | pendiente | Cifras oficiales y validación |

Imágenes Sentinel-2 usadas (nubes medidas **dentro de la zona del incendio** con la capa SCL):

| Incendio | Antes | Después |
|---|---|---|
| Livitaca | 07/10 (0 %) | 17/10 (0 %) |
| Huanoquite | Compuesto: 10/10 (28 % nubes), rellenado con 07/10 (4 %) y 05/10 (0 %) | 17/10 (0,2 %) |

**Por qué no la selva.** Río Tambo (Junín) fue el incendio más grande de la semana, con 1 927 ha
de huella VIIRS. Pero después del fuego, sumando dos imágenes, solo se ve despejado el 33 % de
la zona. En Pucallpa (Manantay, Yarinacocha) las nubes cubren entre 56 % y 99 %. Es temporada de
lluvias: VIIRS (térmico, varias pasadas al día) detecta el fuego, pero Sentinel-2 (óptico) no
puede medir la cicatriz.

## 4. Hallazgos

1. **El fuego grande no es el que toca a más gente.** Hubo 107 incendios con huella > 100 ha en
   77 distritos, y **337 069 personas** viven a 5 km de alguno. El más grande, Río Tambo, tiene
   unas 1 380 personas a 5 km. Uno de 211 ha junto a la ciudad de Abancay tiene 51 429, y la
   periferia de Pucallpa suma unas 75 000 (Yarinacocha 40 765 + Manantay 34 502).
2. **Lo que se quemó en el Cusco es pastizal, no bosque.** Cada incendio se reporta como un rango:
   de dNBR ≥ 0,27 (conservador) a dNBR ≥ 0,10 (incluye la quema leve del pasto seco).

   | Incendio | Pastizal | Bosque | Matorral y cultivo | Total |
   |---|---|---|---|---|
   | **Livitaca** (Chumbivilcas) | **425 – 927 ha** (99,9 %) | 0 ha | ~1 ha | 425 – 929 ha |
   | **Huanoquite** (Paruro) | **381 – 826 ha** (92–93 %) | **24 – 57 ha** (6 %) | 4 – 12 ha | 409 – 895 ha |

   Es coherente con la quema de pastos de la ganadería altoandina, una práctica para provocar el
   rebrote que se escapa de control en plena época seca. Solo en Huanoquite el fuego entró a
   bosque: las franjas de quebrada del sector oeste.
3. **El número depende de las decisiones, y el método estándar subestima el fuego en la puna.** El
   umbral 0,27 (USGS) se calibró en bosques. El pasto seco de octubre parte de un NBR bajo y,
   aunque se queme por completo, cae poco. Con 0,27 queda fuera más de la mitad de la cicatriz que
   se ve en la imagen (Livitaca: 425 frente a 929 ha). La línea de 0,10 coincide con la mancha
   visible en falso color. La huella VIIRS tampoco sirve como sustituto: da 267 ha en Livitaca y
   647 ha en Huanoquite.

![dNBR Livitaca y Huanoquite](figuras/03_dnbr_livitaca_huanoquite.png)

![Área quemada según cobertura](figuras/03_mapa_cobertura_quemada.png)

![Personas a 5 km de incendios > 100 ha](figuras/02_mapa_personas_5km.png)

## 5. Limitaciones

- **Un foco de calor no es un incendio:** incluye quemas agrícolas. Por eso se usa el umbral de 100 ha.
- **Las nubes esconden el fuego.** La selva queda fuera del dNBR. Con más pasadas posteriores, o
  sumando Landsat (HLS) o radar (Sentinel-1), se podría reconstruir.
- **Umbral y pastizal.** Se reporta el rango 0,27–0,10. Con 0,10 aumenta el riesgo de contar pasto
  que solo se secó entre las dos fechas; el mapa en falso color sirve de control visual.
- **WorldCover es de 2021**: la cobertura pudo cambiar (nuevas chacras, pasto convertido).
- **WorldPop es un modelo, no un censo.** Reparte la población del censo según luces, caminos y
  cobertura, y puede subestimar a la población rural dispersa.
- La imagen posterior de Huanoquite (17/10) coincide con los últimos focos: parte del incendio
  pudo seguir creciendo después.

## 6. Cómo reproducir

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_datos_mapa_base.ipynb 02_focos_incendios.ipynb 03_raster_cusco.ipynb
```

`comun.py` guarda las rutas y las funciones compartidas. `src_nb/` tiene el código de cada
notebook en texto plano.
