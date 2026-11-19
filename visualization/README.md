# Visor 3D multisensor

Aplicación Python (Dash + Plotly) que representa en 3D las detecciones de los sensores
`Delta_*` y las trayectorias GNSS de los drones `DroneTag_*`.

## Ejecución

```bash
pip install -r requirements.txt
python app.py --data /ruta/csv --port 8050
```

Abrir http://127.0.0.1:8050

### Con el tracker: trayectoria estimada frente a la real

Tras `python -m src.cli etl`, `track` (y opcionalmente `labels`) en la raíz del repo:

```bash
python app.py --iteration "../data/raw/test/Scenario 1/Iteration 1" \
              --interim ../data/interim --tracks ../reports/tracks.parquet
```

- `--iteration` lee la carpeta original de una iteración (`SensorData/` y `GroundTruthData/`), sin copiar CSV. La iteración (`test:S1:I1`) se deduce de la ruta; si no, `--split-group`.
- `--interim` da el origen ENU, el GT y las etiquetas del ETL.
- `--tracks` es la salida del tracker. `--tracks-b otro/tracks.parquet` carga un segundo run para comparar (antes/después de un cambio).

Qué añade:

| En el panel lateral "Tracker y comparación" | Qué se ve |
|---|---|
| Trayectorias estimadas | cada track del tracker como línea discontinua de su color, junto a la trayectoria real (continua) |
| Error real↔estimado | línea blanca entre cada dron y el track que lo sigue en ese instante; el hover da los metros |
| Elipse de incertidumbre 2σ | zona horizontal donde el tracker cree que está el dron (de `C_nn`, `C_ne`, `C_ee`) |
| Detecciones por etiqueta | cada detección con posición del color del dron al que la asignó `labels`; gris = ruido |
| Run A / B | aparece si se pasa `--tracks-b` |

Panel inferior:

- **Error y cobertura**: error en metros por dron a lo largo del tiempo (con el corte GOSPA), qué track sigue a cada dron (cada color de barra es un track; las marcas rojas son segundos sin track) y tracks falsos por instante. La línea vertical es el instante del reproductor.
- **Resumen por dron**: % del tiempo seguido, error medio y p90, tracks usados, cambios de track y retraso hasta el primer track; GOSPA de la iteración desglosado. Con dos runs, tabla A → B en verde/rojo.

Coordenadas: el visor proyecta con su propio origen y el tracker usa el del ETL (`origins.json`). Los tracks se pasan a lat/lon con su origen y se reproyectan con el del visor, así que no hay desplazamiento entre capas. Los cálculos de la comparación están en [`src/compare.py`](../src/compare.py) (mismo criterio que GOSPA del evaluador).

Para compartir sin lanzar el visor, el mismo análisis en HTML para todas las iteraciones de test:

```bash
python scripts/compare_report.py                 # reports/compare_tracks.html
python scripts/compare_report.py --tracks-b otro/tracks.parquet
```

## Funcionalidades

- Gráfico 3D en coordenadas locales ENU (Este/Norte en metros, origen en el sensor NYTF).
- Un color por sensor; la confianza se codifica de dos formas a elegir:
  - Color por sensor con intensidad proporcional a la confianza (apagado = baja, vivo = alta).
  - Escala global rojo→verde con un símbolo distinto por sensor y barra de color.
  - Opcional: tamaño del punto proporcional a la confianza.
- Selector de sensores y de trayectorias de drones (con atajos «todos / ninguno»).
- Reproducción temporal: Play/Pausa, Inicio, Fin, velocidad 1×–40×, modo acumulado o
  ventana deslizante (N segundos) y barra de tiempo arrastrable. Los drones muestran su
  trayectoria hasta el instante actual y un marcador con su posición.
- Filtros por confianza mínima y altura máxima (útil para ocultar ecos de radar a >1 km).
- Hover con hora, clase, subclase, confianza, id de objeto y estado.

## Decisiones sobre los datos

| Aspecto | Tratamiento |
|---|---|
| EGOI · RADAR | Coordenadas en radianes → convertidas a grados. |
| Confianza | Se usa `detectionConfidence` si varía; si es constante (EGOI, fusión, cyber), se usa la confianza de la clasificación. MMJJ no informa confianza (se dibuja como 1 y aparece «n/d» en el hover). |
| Cámara, acústico, RF pasivo (WVFX, EMXG, YXAB, ORFH) | Sólo dan azimut (y elevación): no hay distancia. Se dibuja un rayo desde el sensor y un punto a una distancia configurable. RF pasivo no tiene elevación → rayo horizontal. |
| Datum vertical | Los sensores usan referencias distintas. Al arrancar se emparejan detecciones con las trazas DroneTag (<1 s, <25 m) y se estima el desfase: NYTF y STUG reportan altura relativa (≈ +70 m respecto al elipsoide), EGOI y fusión usan altitud elipsoidal. Con «Altura sobre terreno» activado todo se expresa como altura aproximada sobre el suelo (cota del terreno ≈ 74,5 m elipsoidal, estimada con los drones en tierra). |
| Drones | Se usa `geo_altitude` (elipsoidal GNSS). |

El resumen de estas decisiones se imprime en consola al arrancar.
