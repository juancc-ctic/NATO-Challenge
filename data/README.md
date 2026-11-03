# Base de datos interim

Generada por `python -m src.cli all` desde `/opt/NATO-Challenge/`. Los CSV originales se copian en `data/raw/{test,eval}` y no se modifican.

## Cómo se crea

`python -m src.cli all` recorre los CSV de test y evaluación, normaliza cada sensor a un **esquema común**, calcula ENU local por iteración, deduplica, escribe las tablas Parquet y genera los splits sin fuga. El detalle de cada tratamiento está en la [tabla de tratamientos](#tabla-de-tratamientos).

El ETL es determinista y puede reejecutarse sobre datos de test no vistos (unidades, esquema común, ENU por iteración con el origen de esa misma iteración —guardado en `origins.json` y reutilizado al invertir—, seriales y clases, mapeando categorías desconocidas a `Unknown`), pero cualquier preprocesamiento aprendido (escaladores, imputación, calibración o umbrales) debe ajustarse solo con train y reutilizarse congelado (`transform`/`inverse_transform`, nunca `fit_transform` sobre test), validando el pipeline contra `eval` como pseudo-no-visto antes de confiar en él.

## Checklist ETL: de raw a interim

Esta es la referencia completa de lo que el ETL hace a los CSV originales, en orden. Si cambia el código, hay que actualizar esta tabla. No se modifica nada de `data/raw`.

Orden: `discover_iterations` → `build_*` (detecciones, ground truth y amigos) → `build_origins` → `add_enu` → `deduplicate` → `_summary` → `write_interim`; después se generan los splits y se valida el Excel. El contrato de columnas está en [`src/schema.py`](../src/schema.py) y se documenta en [Esquema común de detecciones](#esquema-común-de-detecciones), [Esquema de `ground_truth.parquet`](#esquema-de-ground_truthparquet) y [Esquema de `friendly_uas.parquet`](#esquema-de-friendly_uasparquet).

### Tabla de tratamientos

| Etapa | Qué hace | Artefacto |
|---|---|---|
| Descubrimiento | Encuentra `Scenario/Iteration`, `SensorData`, `Truth`/`GrountTruthData`, `FriendDroneList` y plantillas, y genera `split_group`; así cada vuelo queda agrupado desde el inicio y luego se puede separar train/validación sin fuga | `inventory.csv` |
| Lectura CSV | Lee en UTF-8 respetando comillas y anidamientos y cuenta filas malformadas, para no perder datos ni romper el pipeline por un CSV imperfecto | `inventory.csv`, `etl_issues.csv` |
| Tiempo | Convierte ISO 8601 a Unix ms UTC y ordena, para tener una clave numérica única, comparable y sin ambigüedad de zona horaria con la que cruzar sensores y calcular `dt`; descarta timestamps inválidos porque no se pueden situar en el tiempo | `ts_ms`, `ts_iso` |
| Sensor | Unifica `sensor_id = platform_code` y mapea el tipo crudo a un vocabulario canónico, para que todos los sensores sean comparables entre sí | `sensor_id`, `sensor_type` |
| Coordenadas | Normaliza `x=lon`, `y=lat`, `z=alt`, convierte EGOI de radianes a grados y deja todo en WGS84, para no mezclar sistemas ni unidades | `lat/lon/alt`, `raw_*`, `sensor_*` |
| Posición | Marca `0,0` como inválido y decide `has_position`/`pos_source` según el tipo de sensor, porque unos dan posición real y otros solo rumbo, y confundirlos metería estados falsos en el tracker | `has_position`, `pos_source` |
| Azimut/altitud | Pasa azimut/elevación de radianes a grados y marca `has_bearing`/`has_alt`, porque no todos los sensores miden lo mismo y el tracker debe saber qué puede usar | `az_deg`, `el_deg`, `has_bearing`, `has_alt` |
| Sigma | Usa el error reportado o un default por sensor, convierte el error de cyber de grados a metros y aplica floor/cap, para tener una incertidumbre homogénea, plausible y con la que pesar cada medida | `sigma_*`, `sigma_source` |
| ENU | Toma como origen la mediana de sensores de la iteración y pasa ECEF → ENU, dejando `u=NaN` sin altitud, para que el tracker trabaje en metros locales estables en vez de grados y sin coordenadas gigantes | `n/e/u`, `origins.json` |
| Clase | Aplana la clasificación jerárquica, la mapea a Fixed/Rotary/Transmitter/Unknown y añade flags dron/no-dron, para filtrar y entrenar con etiquetas consistentes | `class_*` |
| Serial | Extrae el serial de `objectInfo`/`event`/`id`, lo normaliza y lo compara con la lista de amigos, para identificar amigo/enemigo sin depender del formato original | `serial*`, `friendly_serial_match` |
| Metadatos | Extrae modelo, fabricante, protocolo, frecuencia, banda, MACs, Remote ID, evento y track, porque enriquecen identificación y clasificación con información que la posición no da | `model*`, `manufacturer`, `protocol`, `freq_*`, etc. |
| Ground truth | Lee `DroneTag_COLOR_...`, deriva `color`/`is_blue` y usa `geo_altitude`, y calcula velocidad, heading y `flight_id`, para tener la verdad de referencia alineada con las detecciones | `ground_truth.parquet` |
| Amigos | Lee el Excel, normaliza seriales y crea el conjunto por iteración, para poder marcar identidad amiga en inferencia | `friendly_uas.parquet` |
| Dedup/orden | Calcula un SHA1 de la fila completa, se queda con la primera copia y ordena de forma determinista, para eliminar duplicados exactos y que el resultado sea reproducible | `row_uid`, Parquet |
| Filtros | Aplica `keep_non_drone`, copia CSV opcional y compresión, para conservar lo útil sin inflar el interim | CSV/Parquet |
| Splits | Asigna `test` → dev y `eval` → holdout y hace LOO por iteración, para validar en vuelos no vistos y evitar fuga de información | `splits.json`, `split_*.csv` |
| Check | Valida la cabecera de 19 columnas y copia los timestamps de la plantilla, para garantizar que el Excel de entrega respeta el formato oficial | `check.json`, `reports/check/*.xlsx` |

### Dónde mirar cada cosa

| Pregunta | Artefacto |
|---|---|
| ¿Qué ficheros hay y cuántas filas tiene cada uno? | `reports/inventory.csv`, `reports/sensor_file_summary.csv` |
| ¿Qué se descartó y por qué? | `reports/etl_issues.csv`, `reports/etl_summary.json` (`rows_dropped`, `issues_by_type`) |
| ¿Qué columnas tienen nulos? | `reports/null_counts.csv` |
| ¿Cuántas detecciones, GT y amigos hay? | `reports/etl_summary.json` (`counts`) |
| ¿Cómo queda cada columna y de qué tipo es? | `data/interim/schema.json` |
| ¿Cuál es el origen ENU de cada iteración? | `data/interim/origins.json` |
| ¿Qué folds hay y sin fuga? | `data/interim/splits.json`, `reports/split_manifest.csv`, `reports/split_folds.csv` |
| ¿Las plantillas Excel están bien? | `reports/check.json`, `reports/check/*.xlsx` |
| ¿Dónde están los datos normalizados? | `data/interim/*.parquet`, `data/interim/{test,eval}/*.parquet` |

## Consideraciones en la creación de la BBDD interna

- **`fusion` (`LJJX`)**: solo existe en test y duplica tracks entre `Charlie_LJJX` y `Delta_LJJX`. Para eval hay que excluirlo o deduplicarlo; para train, decidir si se usa.
- **Filas cyber sin `alt_m`**: ya tienen `n`/`e` en 2D y `u=NaN`; el tracker debe aceptar 2D.
- **Filas con sensor `0,0`**: marcadas `has_sensor_location=False` y excluidas del origen/triangulación.
- **`sensor_lat/lon` variables** en algunos sensores: documentado; no asumir constante, usar `has_sensor_location`.
- **`serial` inferido de `objectInfo`/`id`**: conviene revisar falsos positivos/negativos antes de identificación.
- **`friendly_uas` solo eval**: en test no hay lista de amigos (`friendly_available=False`), pero el ground truth de test sí trae `color`/`is_blue` y sirve para entrenar identidad amigo/enemigo.
- **`out/` vacío** hasta la fase de submission; no hay tests automáticos, solo `check`.

Los recuentos exactos (detecciones, GT, amigos, duplicados, plantillas y folds) están en `reports/etl_summary.json` y `reports/check.json`; no los fijes aquí para que no envejezcan.

## Esquema común de detecciones

Una fila por detección:

| Columna | Qué es |
|---|---|
| `ts_ms` | timestamp Unix en milisegundos UTC |
| `sensor_id` | identificador estable del sensor, p. ej. `Delta_EGOI` |
| `sensor_type` | tipo canónico: `radar`, `rf`, `acoustic`, `camera`, `cyber`, `fusion`, `other` |
| `lat` | latitud del objeto en grados WGS84 |
| `lon` | longitud del objeto en grados WGS84 |
| `alt_m` | altitud del objeto en metros; si falta pero hay lat/lon, `n`/`e` se calculan en 2D y `u` queda `NaN` |
| `n` | posición norte ENU local en metros |
| `e` | posición este ENU local en metros |
| `u` | posición vertical ENU local en metros |
| `sigma_n` | incertidumbre norte en metros |
| `sigma_e` | incertidumbre este en metros |
| `sigma_u` | incertidumbre vertical en metros |
| `az_deg` | azimut sensor → objeto en grados |
| `el_deg` | elevación sensor → objeto en grados |
| `has_position` | `True` si la posición es válida para tracking |
| `class_raw` | clase original del sensor |
| `class_conf` | confianza de la clase, 0–1 |
| `serial` | serial del dron o emisor |
| `model` | modelo del dron |
| `protocol` | protocolo de radio |
| `freq_mhz` | frecuencia central en MHz |
| `source_file` | nombre del CSV original |

## Columnas de procedencia (no son features)

| Columna | Qué es |
|---|---|
| `split` | `test` (con GT) o `eval` (sin GT) |
| `dataset` | `Testing`, `Dataset I`, `Dataset II` o `Dataset III` |
| `scenario` | número de escenario |
| `iteration` | número de iteración |
| `split_group` | grupo de split, p. ej. `test:S1:I1`; se usa para folds y evitar fuga |
| `source_relpath` | ruta relativa del CSV original |
| `raw_index` | índice de la fila dentro del CSV original |
| `row_uid` | SHA1 de `split_group + sensor_id + contenido`; identifica y deduplica la detección |

No usarlas como features del modelo: solo para agrupar, unir, deduplicar y auditar.

## Columnas extra de `detections.parquet`

El esquema común es el contrato mínimo. `detections.parquet` añade estas columnas:

| Grupo | Columnas | Qué es |
|---|---|---|
| Tiempo y sensor | `ts_iso`, `sensor_type_raw`, `sensor_code`, `platform`, `sensor_node_id` | timestamp ISO, tipo/código original del sensor, plataforma y `nodeId` |
| Posición del sensor | `sensor_lat`, `sensor_lon`, `sensor_alt_m`, `sensor_n`, `sensor_e`, `sensor_u`, `has_sensor_location` | posición del sensor en WGS84 y ENU; la bandera indica si es válida |
| Posición reportada | `raw_lat`, `raw_lon`, `raw_alt_m`, `raw_n`, `raw_e`, `raw_u`, `coord_system_raw`, `datum_raw`, `pos_source` | posición original antes de decidir `has_position`, sistema de coordenadas y origen |
| Disponibilidad | `range_m`, `has_bearing`, `has_alt`, `has_class`, `has_serial`, `has_velocity`, `has_metadata` | distancia si existe y banderas de campos disponibles |
| Clasificación | `class_norm`, `class_path`, `class_top`, `class_is_drone`, `class_is_non_drone`, `class_is_transmitter` | clase normalizada, ruta jerárquica y flags dron/no-dron/transmisor |
| Metadatos | `serial_norm`, `manufacturer`, `freq_band`, `det_conf` | serial normalizado, fabricante, banda y confianza de detección |
| Velocidad | `vn`, `ve`, `vu`, `sigma_vn`, `sigma_ve`, `sigma_vu` | velocidad ENU y su error |
| RF/cyber | `drone_mac`, `ap_mac`, `pilot_mac`, `remote_id`, `heading_deg`, `channel`, `event_id`, `event_serial`, `event_factory`, `event_takeover` | MACs, Remote ID, rumbo, canal y datos del evento |
| IDs originales | `object_id`, `task_id`, `report_id`, `track_id_raw`, `colour`, `state` | identificadores originales del sensor y estado |
| Flags de uso | `is_fusion_node`, `is_test_only`, `gt_available`, `friendly_available`, `friendly_serial_match` | nodo de fusión, solo-test, GT disponible y coincidencia con lista de amigos |
| Origen de datos | `sigma_source`, `alt_source` | de dónde salen `sigma_*` y `alt_m` |

## Nota sobre `sensor_lat` / `sensor_lon`

Deberían ser constantes por sensor, pero no siempre lo son:

- Variaciones de ~1e-5° son ruido de redondeo y no afectan; variaciones de metros o decenas de metros (`Delta_EGOI`, `Delta_NYTF`, `Delta_IKJX`, `Charlie_MMJJ`, `Charlie_ILYS`) indican sensor móvil o posición estimada y no se deben asumir constantes.
- `Charlie_VKYY` y `Echo_QTVS` (tipo `OTHER`) traen `0.0, 0.0, 0.0`: es un placeholder, no una posición real; se marcan `has_sensor_location=False` y se excluyen del origen ENU y de la triangulación.
- Influye en el origen ENU y en la triangulación de sensores solo-azimut; no en radar/cyber (ya traen posición del objeto) ni en clasificación, identidad o metadatos.

## Esquema de `ground_truth.parquet`

Una fila por segundo y dron. `drone_key` no está en detecciones: está aquí y en `*_ground_truth.parquet`.

| Columna | Qué es |
|---|---|
| `ts_ms` | timestamp Unix en milisegundos UTC |
| `ts_iso` | timestamp ISO UTC |
| `drone_key` | clave única del track GT (`split_group:nombre_fichero`); no es serial; agrupa las filas de un vuelo GT |
| `color` | `RED` o `BLUE` |
| `is_blue` | `True` si el dron es azul/amigo |
| `model_raw` | modelo tal cual del nombre de fichero |
| `model_norm` | modelo normalizado |
| `manufacturer` | fabricante inferido |
| `lat`, `lon` | posición GT en grados WGS84 |
| `alt_m` | altitud GT en metros, desde `geo_altitude` |
| `altitude_raw` | columna `altitude` original, conservada pero no usada |
| `height_m` | altura sobre el punto de despegue |
| `n`, `e`, `u` | posición ENU local |
| `sigma_n`, `sigma_e`, `sigma_u` | incertidumbre de posición |
| `velocity_x`, `velocity_y`, `velocity_z` | velocidad GT |
| `speed_mps` | módulo de velocidad |
| `heading_deg` | rumbo |
| `horizontal_accuracy`, `vertical_accuracy`, `speed_accuracy` | precisiones GT |
| `flight_id` | ID de vuelo Dronetag |
| `alt_source` | `geo_altitude` o `altitude_fallback` |
| `split`, `dataset`, `scenario`, `iteration`, `split_group` | procedencia |
| `source_file`, `source_relpath`, `raw_index` | fichero y fila original |

## Esquema de `friendly_uas.parquet`

| Columna | Qué es |
|---|---|
| `split`, `dataset`, `scenario`, `iteration`, `split_group` | procedencia |
| `manufacturer` | fabricante del dron amigo |
| `serial_raw` | serial tal cual |
| `serial_norm` | serial normalizado para comparar |
| `source_file`, `source_relpath`, `raw_index` | fichero y fila original |

## Artefactos y salidas

| Artefacto | Contenido |
|---|---|
| `data/interim/detections.parquet` | una fila por detección: tiempo, sensor, posición, error, clase, serial y metadatos |
| `data/interim/ground_truth.parquet` | una fila por segundo y dron: referencia Dronetag |
| `data/interim/friendly_uas.parquet` | seriales amigos de evaluación (solo tabla global) |
| `data/interim/{test,eval}/*.parquet` | detecciones y ground truth por iteración |
| `data/interim/origins.json` | origen ENU local de cada iteración |
| `data/interim/splits.json`, `reports/split_manifest.csv`, `reports/split_folds.csv` | folds leave-one-iteration-out sin fuga |
| `data/interim/schema.json` | dtypes exactos de cada tabla |
| `reports/inventory.csv` | auditoría de ficheros, filas, coordenadas y metadatos |
| `reports/sensor_file_summary.csv` | resumen por fichero de sensor |
| `reports/etl_summary.json` | recuentos, calidad, duplicados, descartes e incidencias |
| `reports/etl_issues.csv` | incidencias por fila |
| `reports/null_counts.csv` | nulos por tabla y columna |
| `reports/check.json`, `reports/check/*.xlsx` | validación de plantillas Excel |
| `out/` | solo Excel de entrega |
