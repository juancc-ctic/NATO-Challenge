# Informe breve: modelos recomendados por tarea

| Tarea | Peso | Enfoque recomendado | Por qué |
|---|---|---:|---|
| Tracking | 40% | Kalman de velocidad constante (EKF/UKF) + asociación húngara con gating χ² + ciclo de vida `tentative/confirmed/coast` | Los datos son ruidosos y dispersos; un filtro clásico es robusto, interpretable y se ajusta con pocos parámetros. |
| Clasificación | 20% | Gradient boosting tabular (LightGBM/XGBoost) por detección + voto ponderado por track con histéresis | Hay features tabulares (sensor, cinemática, RF/cyber, confianza); GBM gana a redes profundas con estos volúmenes. |
| Identificación | 10% | Regla de seriales contra la lista de amigos + modelo pequeño solo para casos sin serial | El serial es la señal más fiable; no hace falta un modelo grande. |
| Metadatos | 30% | Reglas de parseo con prioridad cyber/RF + GBM para campos faltantes (modelo, protocolo, banda) | Gran parte es lectura y mapeo de campos, no predicción; el GBM solo rellena huecos. |

## Cómo hacerlo

- Empieza por el tronco: ETL → prefiltro → asociación → tracker. Sin tracks no hay dónde poner el 60% restante.
- Usa validación leave-one-iteration-out y congela escaladores/umbrales antes de `eval`.
- Para clasificación, compara primero regresión logística/Random Forest como baseline; si el GBM no mejora, quédate con lo simple.
- Para identidad y metadatos, prioriza reglas y cobertura; un `Unknown` honesto puntúa más que un dato inventado.
- Si sobra tiempo y hay ventanas de track suficientes, prueba un 1D-CNN/GRU sobre secuencias de features del track; no lo uses como primer modelo.
- Calibra umbrales con *split conformal* usando folds de test para fijar el FPR objetivo y reporta FPR/TPR.
- Evalúa con GOSPA, recall de clase/identidad y acierto de metadatos; guarda todo en `reports/`.

En una frase: **Kalman + húngaro para tracking, GBM tabular + voto por track para clasificación, reglas para identidad/metadatos, y calibración conformal con LOO**.

## Tracking (40%): estimación, fusión multisensor y seguimiento robusto de objetivos.

Este bloque pertenece a **Tracking**, porque define el modelo de medida (`h`, `H`, `R`) y la observabilidad que usa el EKF del tracker.
Tracking, asociación de detecciones y ciclo de vida de los tracks

### Prefiltro de detecciones

Antes del tracker se aplica un prefiltro implementado en `src/prefilter.py`, cuya función genérica `preprocess_detections` delega en un método concreto (`range` o `none`). La limpieza común elimina nodos de fusión, detecciones no-dron y medidas sin posición ni azimut utilizables. El método `range` descarta ecos de posición fuera del radio del escenario, $r_k=\sqrt{n_k^2+e_k^2}\leq r_{\max}$, con $r_{\max}$ en `tracker.prefilter.max_range_m`; las medidas solo-azimut se conservan. El prefiltro podría ampliarse con outliers de saltos de velocidad o aceleración, altura máxima, confianza mínima y filtros por clase o sensor. Es un paso de limpieza de datos, no de estimación.

### Fusión multisensor y análisis de observabilidad geométrica

_(EKF, modelos de medición, matriz de información y selección de sensores)._

El sistema propone estimar la posición tridimensional del dron en coordenadas ENU ($\mathbf{x}=[n,e,u]^T$) mediante un **filtro de Kalman extendido (EKF)**, capaz de fusionar sensores que proporcionan mediciones de distinta naturaleza. Cada sensor se modela mediante una función de observación $\mathbf{z}=h(\mathbf{x})+\mathbf{v}$, cuyo Jacobiano $H=\partial h/\partial\mathbf{x}$ se obtiene derivando dicha función, mientras que $R$ representa la covarianza del ruido de las mediciones. Por ejemplo, un sensor de posición tiene $H=I_3$, mientras que uno de azimut tiene $H=[-\Delta e/r^2,\Delta n/r^2,0]$, donde $\Delta n,\Delta e$ representan la diferencia de posición entre el dron y el sensor. Así, cada sensor aporta información diferente en función de qué magnitudes mide y con qué precisión.

Para evaluar la calidad de la información conjunta de los sensores, se propone calcular la **matriz de información** $J=\sum_i H_i^TR_i^{-1}H_i$. Su rango permite determinar si la geometría de los sensores proporciona información suficiente para estimar las tres dimensiones: $\operatorname{rank}(J)=3$ indica observabilidad completa, mientras que un rango inferior señala direcciones que no pueden estimarse adecuadamente. También pueden analizarse sus valores propios, el número de condición $\kappa=\lambda_{\max}/\lambda_{\min}$ y el incremento de $\log\det(J)$ al incorporar nuevos sensores, permitiendo identificar geometrías débiles, sensores redundantes y aportaciones de información marginales. La incertidumbre de la estimación está relacionada con la inversa de esta matriz, $P\approx J^{-1}$, cuando la matriz es invertible y bajo las hipótesis del modelo linealizado.

Aunque el EKF ya utiliza $H$ y $R$ para fusionar las mediciones y actualizar la estimación mediante $K=P^-H^T(HP^-H^T+R)^{-1}$, no incorpora necesariamente un diagnóstico explícito de la geometría ni una selección inteligente de sensores. Por ello, se propone desarrollar una **capa transversal de prefiltro, selección y diagnóstico multisensor**, que reutilice los mismos $H$ y $R$ del EKF para elegir qué sensores conviene incorporar antes del tracker, actuar como criterio geométrico complementario en el gating, detectar redundancias y advertir cuándo la posición estimada presenta direcciones débilmente observadas. Esta capa no sustituye ni duplica el filtro: el EKF proporciona la estimación y la capa adicional evalúa su fiabilidad y la calidad de la información disponible.

Finalmente, se plantea extender el sistema de tracking mediante criterios de gating alternativos o complementarios al umbral estadístico $\chi^2$, incorporando propiedades algebraicas y geométricas como el rango de las matrices de información, la independencia lineal entre sensores y la consistencia física de las trayectorias. Asimismo, se propone investigar la incorporación de regularización y sparsity en la función de coste del algoritmo húngaro, penalizando asociaciones poco fiables o innecesarias e introduciendo costes de no asociación. De esta manera, se combinarían la estimación dinámica mediante Kalman, la asociación de detecciones, el diagnóstico de observabilidad y la selección de sensores para mejorar la robustez del tracking en escenarios con mediciones ruidosas, redundantes o incompletas.

### Tracking multiobjeto y asociación inteligente de detecciones

_(Kalman, gating χ², algoritmo húngaro, regularización y ciclo de vida de los tracks)._

El **tracking** consiste en seguir la trayectoria de distintos objetos a lo largo del tiempo a partir de mediciones que pueden ser ruidosas, incompletas o dispersas. Para ello, se propone utilizar un filtro de Kalman de velocidad constante, que estima la posición y velocidad de cada objeto mediante un modelo de estado. Por ejemplo, para un movimiento 2D, el estado se define como $\mathbf{x}_k=[x_k,y_k,v_{x,k},v_{y,k}]^T$, y evoluciona según $\mathbf{x}_k=F\mathbf{x}_{k-1}+\mathbf{w}_k$, donde $F$ modela el movimiento y $\mathbf{w}_k$ representa el ruido del modelo. El filtro alterna dos etapas: predicción, $\hat{\mathbf{x}}_{k|k-1}=F\hat{\mathbf{x}}_{k-1|k-1}$, y actualización, que combina la predicción con las nuevas mediciones mediante la ganancia de Kalman $K_k$. Así se obtiene una estimación más robusta que utilizando directamente las detecciones del sensor.

Cuando existen múltiples objetos y detecciones, es necesario resolver el problema de **asociación**, es decir, determinar qué medición corresponde a cada trayectoria. Para ello se propone utilizar el algoritmo húngaro (Hungarian Algorithm), que busca minimizar el coste global de asignación: $\min_{\pi}\sum_i c_{i,\pi(i)}$, donde $c_{ij}$ representa el coste de asociar el track $i$ con la detección $j$. Antes de realizar esta asignación se aplica un mecanismo de **gating**, cuyo objetivo es descartar asociaciones físicamente o estadísticamente improbables. El enfoque clásico utiliza el gating $\chi^2$, basado en la distancia de Mahalanobis: $d^2=\mathbf{r}^TS^{-1}\mathbf{r}$, donde $\mathbf{r}=\mathbf{z}-H\hat{\mathbf{x}}_{k|k-1}$ es el residuo entre la medición y la predicción, y $S=HP_{k|k-1}H^T+R$ es la covarianza de la innovación. Una asociación se considera candidata si $d^2\leq\chi^2_{m,\alpha}$. Por ejemplo, para mediciones 2D y un nivel de confianza del 95 %, el umbral es 5.991. De este modo, el gating determina qué asociaciones son posibles y el algoritmo húngaro selecciona la combinación óptima entre ellas.

Como posible mejora, se propone investigar **criterios de gating alternativos o complementarios al****$\chi^2$**, incorporando propiedades algebraicas, geométricas y multisensor de las mediciones. Entre ellas destacan el rango de las matrices de observabilidad, para comprobar si el sistema dispone de información suficiente para estimar el estado; la independencia lineal entre sensores, para identificar mediciones redundantes o complementarias; y la consistencia física de las trayectorias, utilizando restricciones de velocidad, aceleración o continuidad del movimiento. Estos criterios permitirían descartar asociaciones incompatibles no solamente por su distancia estadística, sino también por la estructura y coherencia de la información disponible. Es importante distinguir que el rango y la independencia lineal evalúan principalmente la observabilidad y la calidad de la información, por lo que pueden complementar al gating sin sustituir necesariamente el criterio estadístico.

También se plantea estudiar la incorporación de **regularización y sparsity en la función de coste del algoritmo húngaro**, favoreciendo soluciones con asociaciones fiables y evitando asignaciones innecesarias. Una formulación conceptual sería $J(X)=\sum_{i,j}C_{ij}X_{ij}+\lambda\|X\|_0$, donde $X_{ij}$ indica si existe una asociación, $C_{ij}$ representa su coste y $\lambda$ controla la penalización. No obstante, dado que el algoritmo húngaro ya impone restricciones de asignación uno a uno, una penalización uniforme sobre el número de asociaciones puede no modificar la solución. Por ello, sería más interesante incorporar nodos ficticios (dummy assignments), costes de no asociación y penalizaciones diferenciadas para controlar cuándo resulta preferible no asignar una detección a ningún track. Esto permitiría adaptar la asociación a escenarios con falsas alarmas, detecciones incompletas y múltiples sensores.

Finalmente, se implementaría un **ciclo de vida de los tracks (track lifecycle)** para gestionar la aparición, evolución y desaparición de los objetos. Se distinguen tres estados principales: _tentative_, cuando una detección genera un nuevo track provisional que todavía debe validarse; _confirmed_, cuando acumula suficientes detecciones coherentes, por ejemplo $N_{\text{hits}}\geq 3$, y se considera un objeto real; y _coast_, cuando deja de recibir mediciones temporalmente, pero se mantiene mediante la predicción del filtro de Kalman sin realizar actualización. Durante este último estado aumenta la incertidumbre de la estimación, por lo que, si el número de observaciones perdidas supera un umbral $N_{\text{misses}}>N_{\max}$, el track se elimina. En conjunto, Kalman estima el movimiento, el gating filtra asociaciones, Hungarian resuelve la asignación y el ciclo de vida gestiona la existencia de cada trayectoria.

### Gating

- **PMBM/GM-PHD** en lugar de húngaro greedy: modela tracks no detectados y falsos; es el estado del arte para GOSPA.
- **Gating conforme + VB-AKF**: radio de asociación calibrado con residuos de validación y `R` estimada online; robusto a colas pesadas y sensores heterogéneos.
- **Covariance intersection**: fusión consistente de tracks con correlaciones desconocidas, sin doble conteo.

## Plus innovador y robusto en las demás tareas

Principio: **no asumir normalidad ni independencia**. Usar métodos *distribution-free* (conformal), fusión con incertidumbre (CI/DS) y calibración con abstención.

### Clasificación (20%)

- **Conformal/APS sobre LightGBM**: conjuntos de clases con cobertura y abstención a `Unknown`; controla el error sin supuesto gaussiano.
- **CRF/HMM sobre el track**: consistencia temporal probabilística en lugar de voto independiente.
- **Calibración (temperature/isotonic)**: probabilidades comparables entre sensores y ECE medido.

### Identificación (10%)

- **Fellegi-Sunter + fuzzy matching**: seriales con errores de formato u OCR y umbral calibrado.
- **Conformal risk control**: FPR de `Hostile` acotado con abstención a `Unknown` y costes asimétricos.

### Metadatos (30%)

- **Dempster-Shafer**: fusión de fuentes conflictivas con ignorancia explícita.
- **Factor graph con restricciones**: relaciones duras (protocolo↔banda, fabricante↔modelo) e inferencia MAP.
- **Conformal multi-etiqueta**: cobertura por campo y abstención a `Unknown`/`Not applicable`.

### Cómo demostrar que es robusto

- **Conformal risk control**: acota el riesgo esperado (p. ej. FPR ≤ 5 %) con garantía de muestra finita; calibrar con LOO y congelar antes de `eval`.
- **DRO/group DRO + importance weighting**: robustez ante cambio de distribución entre iteraciones y sensores, reportando cobertura empírica, ECE y FPR/TPR con intervalos.

En una frase: **PMBM + gating conforme + CI para tracking, LightGBM + CRF + conformal para clasificación, Fellegi-Sunter + CRC para identidad, y Dempster-Shafer + factor graph + conformal para metadatos**.

## Postfiltro opcional

Además del tracker, se implementa un postfiltro opcional en `src/postfilter.py`, desactivado por defecto (`tracker.postfilter.method: none`) y aplicado después del remuestreo, sobre la rejilla de salida. Sus métodos son: relleno de huecos por interpolación lineal hasta `max_gap_ms`, suavizado por media móvil de ventana `smooth_window`, y descarte de tracks cuya velocidad supere `max_speed_mps`. El relleno de huecos es distinto del remuestreo: el remuestreo, implementado en `src/resample.py`, predice con `F` y `Q` para llevar los tracks a la rejilla de entrega, mientras que el postfiltro interpola observaciones. Una mejora futura es el suavizado RTS sobre el estado completo, que requiere guardar velocidad y covarianzas de predicción y debería ejecutarse antes del remuestreo.

## Referencias

- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Rahmathullah, A. S., García-Fernández, Á. F., & Svensson, L. (2017). Generalized Optimal Sub-Pattern Assignment Metric. *20th International Conference on Information Fusion (FUSION)*, 1–8.
- Bar-Shalom, Y., Li, X.-R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley. Cap. 6: modelo de velocidad constante y ruido de proceso DWNA.
- Blackman, S. S., & Popoli, R. (1999). *Design and Analysis of Modern Tracking Systems*. Artech House. Cap. 6.
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press. Cap. 4 y 9.
- ResearchGate. *Single-target state estimation using Kalman filter and RTS smoother*. https://www.researchgate.net/figure/Single-target-state-estimation-using-Kalman-filter-and-RTS-smoother_fig4_320182941
