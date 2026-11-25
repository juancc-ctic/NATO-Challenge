# Guía del visor 3D multisensor

Herramienta para **ver qué detectan los sensores, dónde están los drones reales (GT) y qué trayectorias genera nuestro tracker**, y medir cuánto se parecen. Sirve para depurar el pipeline antes de entregar; no genera la entrega.

## 1. Puesta en marcha

```bash
cd visualization
pip install -r requirements.txt
python app.py --iteration "../data/raw/test/Scenario 1/Iteration 1" \
              --interim ../data/interim --tracks ../reports/tracks.parquet
```

Abrir http://127.0.0.1:8050. Los datos y los tracks se cargan al arrancar (unos segundos).

Antes hay que haber ejecutado en la raíz del repo: `python -m src.cli etl`, `track` y, opcional, `labels` (etiquetado) y `submit` (Excel de entrega).

### Argumentos

| Argumento | Para qué | Sin él |
|---|---|---|
| `--iteration <carpeta>` | Carpeta de una iteración tal cual viene del reto (`SensorData/` + `GroundTruthData/`). Deduce `test:S1:I1` de la ruta | Hay que usar `--data` con CSV sueltos |
| `--interim <carpeta>` | Salida del ETL: origen ENU, GT, etiquetas | No hay capa de tracker |
| `--tracks <parquet>` | Salida del tracker | Solo sensores y GT |
| `--tracks-b <parquet>` | Segundo run para comparar A/B | Sin comparación |
| `--split-group` | Fijar `test:S1:I1` si no se deduce de la ruta | — |
| `--data-root` | Carpeta raíz con todas las iteraciones (para el selector) | Se deduce de `--iteration` |
| `--submission` | Carpeta del Excel de entrega | `<repo>/out/` |
| `--config` | `config.yaml` alternativo | El del repo |
| `--port` | Puerto | 8050 |

> Con solo `--iteration` salen únicamente sensores y GT. Para ver tracks hacen falta `--interim` **y** `--tracks`.

## 2. Pantalla

- **Centro**: gráfico 3D (ENU, metros; Este/Norte/Altura). Ratón: arrastrar rota, rueda zoom, botón derecho desplaza. El hover da el detalle de cada punto.
- **Panel lateral**: todas las opciones, en secciones.
- **Barra inferior**: reproductor temporal.
- **Panel inferior** (solo si hay GT y tracks): gráficas de error/cobertura y tablas de resumen.
- **Contador** (arriba): cuántos puntos/tracks se están dibujando.

## 3. Panel lateral

### 3.1 Iteración
Desplegable con todas las iteraciones de la carpeta de datos (test y eval). Al cambiar recarga sensores, drones, tracks, etiquetas y Excel sin relanzar (~10 s; la primera vez que se abre una iteración, ~13 s más por leer su Excel, que queda en caché). En eval no hay GT: se ocultan el panel inferior y las métricas.

### 3.2 Sensores y drones
- Listas con casillas y atajos *todos / ninguno*. Con tracks cargados los sensores **arrancan desmarcados** (si no, la escena va lenta); actívalos según lo que quieras mirar.
- Un color por sensor. Radar, cámara, acústico, RF, cyber y fusión se distinguen en la leyenda.
- Sensores **solo de rumbo** (cámara, acústico, RF pasivo): no dan distancia; se dibuja un rayo desde el sensor y un punto a una distancia configurable.
- Sensores sin posición válida (placeholders `(0,0)`) no se dibujan.
- Los drones reales (GT, `DroneTag_*`) aparecen como línea continua con un marcador en su posición actual.

### 3.3 Apariencia y filtros
- **Confianza**: dos modos — color del sensor con intensidad por confianza, o escala global rojo→verde con símbolo por sensor. Opcional: tamaño del punto proporcional.
- **Confianza mínima** y **altura máxima** (útil para ocultar ecos de radar a >1 km).
- **Altura sobre terreno**: expresa todo como altura aproximada sobre el suelo (terreno ≈ 74,5 m elipsoidal).
- **Datum vertical**: NYTF y STUG miden altura relativa (≈ +70 m), EGOI y fusión elipsoidal. El visor estima el desfase emparejando detecciones con los DroneTag y lo corrige; la consola lo resume al arrancar.

### 3.4 Tracker y comparación
Aparece al cargar `--tracks`.

| Opción | Qué dibuja |
|---|---|
| Trayectorias estimadas | Cada track como línea discontinua de su color (el GT es continuo) |
| Error real↔estimado | Línea blanca dron–track que lo sigue en ese instante; el hover da los metros |
| Elipse de incertidumbre 2σ | Zona horizontal donde el tracker cree que está el dron (`C_nn`, `C_ne`, `C_ee`). Tamaño ajustable |
| Detecciones por etiqueta | Cada detección con el color del dron al que `labels` la asignó; gris = ruido. Conviene ocultar los sensores |
| Detecciones sin track cercano (✕) | Ver 3.5 |
| Ocultar tracks de menos de (s) | Deslizador. Oculta tracks cortos (suelen ser ruido) **solo en pantalla**; las métricas siguen usando todos |
| Run A / B | Selector, solo con `--tracks-b` |

### 3.5 Detecciones sin track cercano
Una ✕ (un color por sensor) marca las detecciones que **ningún track tiene dentro de la puerta del sensor** en su instante: lo que el tracker está ignorando. Usa las mismas reglas que `labels` (distancia para sensores con posición, ángulo para los de solo rumbo) y funciona también en eval.

- Radares ocultos por defecto (su ruido es lo que debe ignorarse); "incluir radares" los muestra.
- Tabla por sensor: detecciones, sin track, % sin track y, según el caso:
  - **test** → "Drones sin track": detecciones de un dron real (según el etiquetado) que ningún track recoge.
  - **eval** → "% drones en test": qué parte de lo que detecta ese sensor era dron real en test. Sensor fiable + muchas sin track = drones probablemente perdidos.
- Cálculo en `src/compare.py` (`detections_vs_tracks`).

### 3.6 Excel de entrega
Carga el Excel que genera `python -m src.cli submit` (el que se envía y puntúa). Se busca en `<repo>/out/` por el nombre de su plantilla.

- **Pintar el Excel de entrega**: tracks en líneas rosas; el hover muestra clase e identidad tal cual van en el fichero.
- **Comprobaciones** (✔ / ! / ✖): cabecera de 19 columnas en orden · rejilla de 100 ms sin saltos · filas con track con lat/lon · covarianzas válidas · clase e identidad con valores admitidos · % relleno (≠ Unknown) · posiciones en la zona (<20 km) · coincide con `tracks.parquet` (valida la conversión ENU→WGS84) · recuento de tracks.

## 4. Reproductor
Play/Pausa, Inicio, Fin, velocidad 1×–40× y barra arrastrable. Dos modos de historia: **acumulado** (todo hasta ahora) o **ventana deslizante** de N segundos. Mover el tiempo solo actualiza lo necesario (el panel inferior se refresca sin redibujar todo), y la línea vertical de las gráficas marca el instante actual.

## 5. Panel inferior

**Error y cobertura** (tres filas, mismo eje de tiempo):
1. *Error*: metros por dron, con el corte GOSPA (c = 50 m).
2. *Cobertura*: qué track sigue a cada dron; cada color de barra es un track, las marcas rojas son segundos sin track.
3. *Tracks falsos*: tracks que no siguen a ningún dron, por instante.

**Resumen por dron** (tabla): % del tiempo seguido, error medio y p90, tracks usados, cambios de track y retraso hasta el primer track. Debajo, el GOSPA de la iteración desglosado (localización, falsos, perdidos). Con dos runs aparece A → B en verde (mejora) / rojo (empeora).

## 6. Cómo leer lo que ves

| Observación | Probable causa |
|---|---|
| Muchos trocitos cortos sueltos | Falsos tracks del ruido; prueba el deslizador de duración mínima |
| Líneas verticales | Desfase de altura entre sensores |
| Rectas largas cruzando la escena | Saltos imposibles; lo cubriría el postfiltro `plausibility` (apagado) |
| Dron sin track durante mucho rato | Ver ✕ de sensores fiables; tracks que no llegan a confirmarse con sensores lentos |
| Track bueno pero desplazado ~2–4 s | Latencia de fusión/radar sin corregir |
| Error alto con cobertura buena | Sesgo de posición o de altura |

## 7. Informe estático
Para compartir sin lanzar el visor (HTML con todas las iteraciones de test):

```bash
python scripts/compare_report.py                      # reports/compare_tracks.html
python scripts/compare_report.py --tracks-b otro/tracks.parquet
```

## 8. Decisiones sobre los datos

| Aspecto | Tratamiento |
|---|---|
| EGOI · radar | Coordenadas en radianes → grados |
| Confianza | `detectionConfidence` si varía; si es constante (EGOI, fusión, cyber), la de clasificación; MMJJ no informa (se dibuja como 1, hover «n/d») |
| Posiciones `(0,0)` | Se tratan como ausentes (evita ejes de millones de metros) |
| Coordenadas | El visor proyecta con su propio origen; los tracks (origen del ETL, `origins.json`) se pasan a lat/lon y se reproyectan, sin desplazamiento entre capas |
| Drones | `geo_altitude` (elipsoidal GNSS); clave de dron incluye hora de despegue para distinguir modelos repetidos |
| Cálculo de comparación | `src/compare.py`, mismo criterio GOSPA que el evaluador |

## 9. Problemas habituales

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError: dash/plotly` | `pip install -r visualization/requirements.txt` |
| No salen tracks | Falta `--interim` y/o `--tracks` |
| Responde una versión vieja | Otro visor ocupa el puerto: cerrar el proceso o usar `--port` |
| Escena muy lenta | Dejar sensores desmarcados; subir confianza mínima; ocultar radares |
| Dropdown de iteraciones vacío | Comprobar `--iteration`/`--data-root` apuntan a la carpeta con `test/` y `eval/` |
| Sin panel inferior | Iteración sin GT (eval) o sin tracks |
