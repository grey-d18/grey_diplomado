# Tarea Qwen: reel informativo de 60 s

Reel vertical de no más de 60 segundos sobre los **incendios en la puna del Cusco (octubre 2026)**, hecho a partir de un informe largo, siguiendo este flujo:

| Paso | Qué es | Herramienta prevista | Archivo |
|---|---|---|---|
| 1 | Texto largo (informe, sentencia o plan) | — | `Incendios_puna_Cusco_octubre_2026.pdf` |
| 2 | Guion | Qwen3.8 lo escribe | `salidas/guion.json` |
| 3 | **Revisión humana contra la fuente** | nosotros | `salidas/guion.json` |
| 4 | Tu voz | Qwen3-TTS la clona | `salidas/reel.wav` |
| 5 | Video | Whisper subtitula | `salidas/reel.mp4` |

Los pasos 2, 4 y 5 los hacen modelos abiertos. El 3 no lo hace ninguna máquina.

## Cambio en el paso 2: sin Qwen3.8, por limitaciones de la computadora

Este trabajo se hace en una computadora con **chip Apple A18 Pro, 8 GB de RAM y unos 19 GB libres de disco**.

Qwen3.8 solo se publica en tamaño **27B**. Su versión más liviana en Ollama (`qwen3.8:27b`, cuantizada a 4 bits) pesa **18 GB** y necesita unos 20 GB de memoria para correr. No cabe en 8 GB de RAM, y la descarga casi llenaría el disco. Tampoco existe una versión más pequeña ni una versión en la nube en Ollama.

Por eso **no corremos Qwen3.8 en local**. El paso 2 se hace como indica la clase para "cualquier laptop": `guion.py` arma el prompt, se pega en un chat (chat.qwen.ai, chat.deepseek.com o Claude) y la respuesta vuelve a `guion.py`, que la valida. Los pasos 4 y 5 sí corren con modelos abiertos en esta computadora, en versiones MLX para Apple Silicon que caben en 8 GB.

### `guion.py`

```bash
python3 guion.py --texto Incendios_puna_Cusco_octubre_2026.pdf --solo-prompt > prompt.txt
# pegar prompt.txt en el chat y guardar lo que responda en respuesta.txt
python3 guion.py --texto Incendios_puna_Cusco_octubre_2026.pdf --respuesta respuesta.txt
```

- Funciona con el `python3` del sistema: si falta `pymupdf`, se relanza solo con `.venv`.
- `--respuesta` acepta la respuesta tal como la copia el chat, con o sin marcas ` ```json `.
- Escribe `salidas/guion.json` con `"revisado": false`.
- Marca con `¡¡` toda `cifra` que no aparece tal cual en el texto. También avisa si una `frase` tiene dígitos o si el guion pasa de 140 palabras (unos 58 s).

## Formato de `salidas/guion.json`

```json
{
  "titulo": "...",
  "escenas": [
    {
      "frase": "Cincuenta y seis incendios activos superan las cien hectáreas…",
      "cifra": "56",
      "etiqueta": "Incendios activos > 100 ha"
    }
  ],
  "revisado": false
}
```

- `frase`: lo que se dice en voz alta, con los números en palabras.
- `cifra` y `etiqueta`: lo que se ve en pantalla. La cifra se copia tal cual de la fuente.

## Paso 3: revisión

El código solo verifica números. Las afirmaciones las verifica una persona. Para cada escena:
1. ¿La `cifra` está en el informe? Las que tienen `¡¡` no están: se corrigen o se quitan.
2. ¿La `frase` dice lo mismo que el informe, sin exagerar ni mezclar datos?
3. ¿Se entiende al oírla una sola vez?

Cuando el guion esté aprobado, se cambia `"revisado": false` por `"revisado": true`. Los pasos 4 y 5 no se ejecutan sin esa marca.

### Qué corregimos del guion de Qwen y por qué

Qwen respondió en chat.qwen.ai (`respuesta.txt`). Las 7 cifras coincidían con el informe. Se corrigieron cuatro frases que decían más de lo que el estudio respalda:

| Escena | Qwen escribió | Quedó | Por qué |
|---|---|---|---|
| 1 | «…ardieron en la puna del Cusco este octubre» | «…ardieron en dos incendios del Cusco este octubre» (etiqueta: «Dos incendios, cifra máxima») | 1 824 ha es la suma de las cifras altas de **dos** de los 11 incendios del Cusco, no el total de la región. Además, Huanoquite ardió entre 2 929 y 3 929 m e incluyó bosque y matorral: no todo fue puna. |
| 5 | «…doscientas dieciséis edificaciones… Son familias cuyos pastos se quemaron» | «…doscientas dieciséis edificaciones a menos de un kilómetro, junto a pastos que se quemaron» | Open Buildings cuenta techos (también cocinas, depósitos y corrales), no hogares. |
| 6 | «Para Huanoquite, no hay reporte» | «Para Huanoquite, no se encontró reporte» | La búsqueda se hizo el 21 de octubre: lo comprobado es que no se encontró. |
| 7 | «El satélite mide hasta cuatro coma seis veces más. Así, la quema de la puna queda fuera de las estadísticas» (cifra: «4,6 veces») | «El satélite mide entre dos y casi cinco veces más. Y un incendio en pastizal, sin daños a personas, puede ni siquiera llegar al registro» (cifra: «2,1–4,6 veces») | 4,6 sale de la cifra alta, la de más riesgo de sobrestimar; el informe da el rango 2,1–4,6. La quema de Livitaca sí está registrada, con una cifra menor; la que falta es la de Huanoquite. |

Las escenas 2, 3 y 4 quedaron igual. Guion final: `salidas/guion.json`, 7 escenas y 112 palabras.

## Pasos 4 y 5: voz y video

| Uso | Modelo | Librería |
|---|---|---|
| Clonación de voz | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | `mlx-audio` |
| Subtítulos y control de la voz | `mlx-community/whisper-large-v3-turbo` | `mlx-whisper` |
| Cuadros del video | — | `pillow` |
| Montaje y volumen | — | `ffmpeg` (Homebrew) |

Se usa el Qwen3-TTS de 0.6B (8 bits) en lugar del de 1.7B para dejar margen de memoria en 8 GB.

```bash
cd tarea_qwen
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-audio mlx-whisper pymupdf pillow jiwer num2words soundfile numpy

python3 voz.py --voz mi_voz.m4a --voz-texto mi_voz.txt   # paso 4 → salidas/reel.wav
python3 hacer_video.py                                     # paso 5 → salidas/reel.mp4, reel.srt, portada.png
```

**`voz.py`**
- Solo corre si `salidas/guion.json` tiene `"revisado": true`.
- Necesita una grabación de referencia (15–25 s de voz limpia) y su transcripción exacta (`--voz-texto`). Sin `--voz-texto`, la transcribe Whisper.
- Genera cada escena por separado. Whisper la vuelve a escuchar: si entiende mal más del 25 % de las palabras, o no está seguro de la primera palabra, la genera de nuevo (hasta 3 intentos).
- Quita los silencios de los bordes, suaviza la entrada de cada escena y pone 0,5 s de silencio al inicio. Qwen3-TTS a veces arranca cortando la primera sílaba.
- `--solo 1,6` rehace solo esas escenas.

**`hacer_video.py`**
- 1080 × 1920, 30 fps, H.264 + AAC, volumen normalizado a −14 LUFS (dos pasadas de `loudnorm`).
- La cifra y la etiqueta de cada escena en el centro; el subtítulo debajo, con la palabra que suena en naranja.
- «Voz generada con IA» durante todo el video, y una tarjeta final con las fuentes.
- Todo queda dentro de la zona segura (lejos de los bordes, donde cada app pone sus botones).
- Los subtítulos usan las palabras del guion, bien escritas, con los tiempos que oye Whisper. Así "Livitaca" no aparece como "Libitaca".

## Resultado

| | |
|---|---|
| Duración | 45,8 s (43,3 s de voz y 2,5 s de tarjeta final) |
| Escenas | 7, 112 palabras |
| Palabras mal entendidas por Whisper | 0–14 % por escena; todos los errores son nombres de lugar (Livitaca, Huanoquite) |
| Volumen | −14,2 LUFS |

## Deficiencias y lo que haría distinto

- **El guion es correcto pero plano.** Cumple con no inventar, pero no engancha: abre con una cifra (1 824 ha) en vez de una pregunta o un contraste, y casi todas las escenas repiten la estructura "lugar + número". Lo más llamativo del informe, que el registro oficial casi no ve estos incendios, aparece recién en la escena 6. En el próximo proyecto pediría en el prompt un gancho explícito (pregunta, contraste o dato sorpresa en los primeros 3 segundos) y pondría el hallazgo principal al inicio.
- **Me costó entender el formato JSON y el ida y vuelta con el chat.** Al principio el guion se armó con otro formato (`texto` / `cita_fuente`) y no con el de la clase (`frase` / `cifra` / `etiqueta`). La respuesta del chat llegó envuelta en ` ```json `, y no estaba claro qué archivo era el que leía el paso de la voz. Se resolvió haciendo que `guion.py` limpie la respuesta y escriba un único `salidas/guion.json`, que es el que leen `voz.py` y `hacer_video.py`. Para la próxima: definir el formato antes de escribir el prompt y probar el ciclo completo con un texto corto.
- **No se usó Qwen3.8 en local** por falta de memoria (ver arriba). El guion lo escribió Qwen en chat.qwen.ai.
- **Nombres propios.** Qwen3-TTS pronuncia "Livitaca" y "Huanoquite" de forma algo insegura. Un modelo de 1.7B o una grabación de referencia que incluya esos nombres podría mejorarlo.

## Privacidad: qué no se sube

El repositorio no incluye la voz. Quedan fuera (ver `.gitignore`):
- la grabación original y su transcripción,
- `salidas/reel.wav`, `salidas/escenas/` y `salidas/reel.mp4`, que tienen la voz clonada.

Se suben el código, el guion, la respuesta de Qwen, los subtítulos, la portada y los posts. Con eso se ve el trabajo sin publicar la voz.

## Archivos

| Archivo | Qué es | En el repo |
|---|---|---|
| `Incendios_puna_Cusco_octubre_2026.pdf` | Fuente (paso 1) | sí |
| `guion.py` | Prompt y validación del guion (paso 2) | sí |
| `respuesta.txt` | Lo que respondió Qwen, sin corregir | sí |
| `salidas/guion.json` | Guion revisado (pasos 2 y 3) | sí |
| `voz.py` | Narración con voz clonada (paso 4) | sí |
| `hacer_video.py` | Subtítulos y video (paso 5) | sí |
| `salidas/reel.srt` | Subtítulos aparte | sí |
| `salidas/portada.png` | Portada del reel | sí |
| `salidas/posts.md` | Tres posts (Instagram/Facebook, X, LinkedIn) | sí |
| `prompt.txt` | Prompt para el chat; se regenera con `guion.py` | no |
| `salidas/escenas.json` | Tiempos de cada escena; se regenera con `voz.py` | no |
| grabación de referencia | Tu voz original | no |
| `salidas/reel.wav`, `salidas/escenas/`, `salidas/reel.mp4` | Voz clonada y video | no |
