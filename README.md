# Visor 3D multisensor

Aplicación Python (Dash + Plotly) que representa en 3D las detecciones de los sensores
`Delta_*` y las trayectorias GNSS de los drones `DroneTag_*`.

## Ejecución

```bash
pip install -r requirements.txt
python app.py                  # usa la carpeta ./data
python app.py --data /ruta/csv --port 8050
```

Abrir http://127.0.0.1:8050

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
