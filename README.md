<img width="1915" height="911" alt="dashboard_vista_general" src="https://github.com/user-attachments/assets/7fd5d6a6-3f6f-42aa-9dec-06f34fd10550" />
# 🔥 Predicción de susceptibilidad de incendios forestales en la Comunitat Valenciana

**Pipeline de datos multifuente + LightGBM + explicabilidad SHAP + dashboard en Streamlit.**
Trabajo Fin de Grado en Ciencia de Datos · Universitat de València (ETSE) · Calificación: 8,8

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2C6E49)
![SHAP](https://img.shields.io/badge/SHAP-explicabilidad-FF0051)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A)

<!-- Si despliegas en Streamlit Community Cloud, descomenta esta línea y pon tu URL:
[![Abrir demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://TU-APP.streamlit.app)
-->

![Dashboard](docs/img/dashboard.png)
<!-- Sube una captura o un GIF corto del dashboard a docs/img/ con este nombre -->

> **In English:** End-to-end ML system that estimates daily wildfire ignition susceptibility across the Valencian Community (Spain). It merges 4,476 official fire records (2010–2022) with ERA5 weather reanalysis, SRTM topography and 1:10,000 land-cover maps, trains LightGBM classifiers under two pseudo-absence sampling designs, tunes the decision threshold for civil-protection priorities (F2-score), explains every prediction with SHAP and serves it all in an interactive Streamlit dashboard. **ROC-AUC 0.94 · Recall 89 % at the operational threshold.**

---

## El problema

La Comunitat Valenciana combina veranos largos y secos, relieve abrupto y una masa forestal continua cada vez menos gestionada. Los índices de peligro clásicos (como el FWI canadiense) se calculan sobre variables meteorológicas puntuales y no capturan la interacción entre **clima, relieve y combustible** que determina dónde y cuándo prende un incendio.

**Objetivo:** estimar la probabilidad de ignición en cualquier punto y fecha del territorio, de forma que un técnico de Protección Civil pueda ver *dónde* está el riesgo y *por qué* el modelo lo señala.

## Los datos

Cuatro fuentes públicas e independientes, integradas por coordenada y fecha:

| Dominio | Fuente | Qué aporta |
|---|---|---|
| Incendios | **EGIF** — Estadística General de Incendios Forestales (MITECO) | 4.476 igniciones validadas, 2010–2022 |
| Meteorología | **ERA5** vía API de Open-Meteo | Temperatura máx., viento medio y rachas, humedad, precipitación 7 días, días sin lluvia |
| Topografía | **SRTM 30 m** vía Open Topo Data | Elevación, pendiente y orientación |
| Vegetación | **COSCV 2024** (Generalitat Valenciana, escala 1:10.000) | Tipo de combustible (6 clases) |

Retos de ingeniería de datos resueltos en el pipeline:

- **Coordenadas UTM con husos 30 y 31 mezclados** sin indicador → reproyección a WGS84 según la longitud de cada punto.
- **Superficies corruptas** en grandes incendios (p. ej. `11.904.680.000.000.000` en lugar de 11.904,68 ha) → corrección manual contra los informes oficiales.
- **40 % de nulos en la API de AEMET** → migración a reanálisis ERA5, con cobertura del 100 %.
- **Pendiente y orientación sin ráster local** → método propio de la "cruz de cinco puntos": cinco consultas de elevación separadas ~55 m y cálculo del vector gradiente.

Resultado: dataset maestro de 4.476 incendios con **0 valores nulos**.

## El modelo

**1. Generación de la clase negativa.** El EGIF solo registra lo que ardió, así que hay que fabricar las "no-igniciones" (pseudo-ausencias). Comparé dos diseños con filosofías opuestas:

- **Alternativa A — Caso-control emparejado:** puntos aleatorios en zona forestal con la *misma fecha* que el incendio real.
- **Alternativa B — Muestreo espacio-temporal puro:** puntos aleatorios con *fecha aleatoria* 2010–2022, para recuperar la varianza climática entre estaciones.

Ambos dan un dataset balanceado de 8.952 registros (50/50), con partición estratificada 80/20.

**2. Selección de algoritmo.** Competición *baseline* XGBoost vs. LightGBM → gana LightGBM (AUC 0,937 vs. 0,933 y menos falsos negativos). Optimización con `RandomizedSearchCV` (50 combinaciones × 5-fold estratificado).

**3. Umbral de decisión orientado a Protección Civil.** No detectar un incendio cuesta mucho más que una falsa alarma. En lugar del umbral por defecto (0,50), elegí el umbral que maximiza el **F2-score** (que pondera el recall el doble que la precisión): **0,30**.

**4. Explicabilidad.** `shap.TreeExplainer` sobre todo el conjunto de test: importancia global (beeswarm, bar plot) y explicación individual de cada predicción (waterfall).

## Resultados

Conjunto de test: 1.791 registros no vistos durante el entrenamiento.

| Métrica (umbral 0,30) | Alternativa A | Alternativa B |
|---|---|---|
| ROC-AUC | 0,938 | **0,940** |
| Recall (incendios detectados) | **89,6 %** | 89,1 % |
| F2-score | **0,878** | 0,874 |
| Falsos negativos | 93 | 98 |
| Falsos positivos | 183 | 181 |

**Impacto del umbral (Alternativa A):** pasar de 0,50 a 0,30 reduce los incendios no detectados de **166 a 93** (73 incendios recuperados) a cambio de 106 falsas alarmas adicionales.

### Hallazgos principales

- **Desacople estival temprano:** julio y agosto concentran más igniciones, pero **junio** acumula más hectáreas quemadas. El desbroce preventivo debe estar terminado antes del 1 de junio.
- **La paradoja del viento:** SHAP muestra que el *viento en calma* aumenta la probabilidad de ignición. Mi hipótesis: el EGIF registra el **inicio** del fuego (mayoritariamente de origen humano), y los días de viento fuerte se activan restricciones y vigilancia, mientras que los días de calma concentran quemas y negligencias. El viento sí es determinante para la *propagación*: los Grandes Incendios (≥ 500 ha) se concentran en días de más de 30 °C y rachas de más de 30 km/h.
- **Umbría vs. solana:** hay más igniciones en solana, pero las laderas de umbría acumulan 33.060 ha quemadas frente a 7.252 ha en solana. Más humedad → más biomasa → incendios mayores cuando las condiciones se vuelven extremas.
- **Los dos diseños de muestreo convergen** en métricas y en ranking de variables, lo que indica que los inductores principales del modelo son estables frente al diseño experimental.

### Limitaciones (y cómo las abordaría)

- **Las métricas miden la separación entre incendios reales y puntos aleatorios**, no el rendimiento operativo real en un día cualquiera. Siguiente paso: validación con partición temporal (entrenar hasta 2019, evaluar 2020–2022) y espacial por bloques, para descartar que la autocorrelación espacial infle los resultados.
- **Latitud y longitud están entre las variables más importantes**: el modelo aprende en parte la geografía histórica del riesgo, no solo la física de la ignición.
- **El objetivo es binario (ignición sí/no)**: no predice la severidad. La línea más prometedora es un modelo de regresión sobre la superficie quemada.
- **El dashboard es retrospectivo** (trabaja sobre el histórico de test). Conectarlo a la API de predicción de Open-Meteo lo convertiría en un sistema de alerta a 24–48 h.

## Demo: el dashboard

`src/dashboard.py` (Streamlit + Folium) ofrece:

- **Mapa de riesgo:** cada punto coloreado como seguro / alerta / alerta crítica (alerta en coníferas o matorral).
- **Selector de modelo** (Alternativa A o B), **de umbral** (0,50 · 0,40 · 0,30 · 0,20) y **de mes/año**; las métricas se recalculan al instante.
- **Auditoría SHAP "a la carta":** elige cualquier registro y ve qué variables empujaron la predicción hacia arriba o hacia abajo.
- **Matriz de confusión** del escenario temporal seleccionado.

## Cómo ejecutarlo

### Camino A — Ver el dashboard (≈ 2 minutos)

El repositorio incluye los modelos entrenados (`lgbm_model.pkl`, `lgbm_model_altB.pkl`) y las matrices de entrenamiento y test de ambas alternativas en `datos/processed/`. No hace falta descargar datos ni reentrenar.

```bash
git clone https://github.com/marcoscivera08/tfg-incendios-valencia.git
cd tfg-incendios-valencia

# Opción 1: Conda (recomendado)
conda env create -f environment.yml
conda activate tfg-incendios

# Opción 2: pip
pip install -r requirements.txt

streamlit run src/dashboard.py
```

Se abre en `http://localhost:8501`.

### Camino B — Reconstrucción completa desde los datos originales

Reproduce todo el experimento: descarga, enriquecimiento, entrenamiento y exportación de modelos. Duración aproximada: 2–4 h (depende de la latencia de las APIs; los scripts tienen reintentos y caché en `datos/external/cache_aemet/` para poder reanudar).

**Requisitos previos**

1. **Datos de partida en `datos/raw/`** (no se incluyen por tamaño y licencia):
   - Registro EGIF de incendios forestales 2010–2022 — [datos abiertos del MITECO](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-datos.html)
   - Cartografía COSCV 2024 — [Institut Cartogràfic Valencià](https://icv.gva.es/)
   <!-- TODO: indica aquí los nombres exactos de archivo que esperan los scripts -->
2. **API key gratuita de [AEMET OpenData](https://opendata.aemet.es/centrodedescargas/altaUsuario)**, guardada en `datos/APIs/api_AEMET.txt` (una sola línea con la clave). Esta ruta está en `.gitignore`: **nunca subas tu clave al repositorio**.

ERA5 (Open-Meteo) y SRTM (Open Topo Data) son APIs abiertas y no requieren clave.

**Fase 1 · Pipeline de datos** — ejecutar en este orden; cada script lee la salida del anterior en `datos/processed/`:

| Paso | Comando | Salida |
|---|---|---|
| 1.1 Limpieza EGIF + cruce con estaciones AEMET | `python src/run_pipeline.py` | `03_dataset_final_CV.csv` |
| 1.2 Imputación climática con ERA5 | `python src/completar_clima_era5.py` | `04_dataset_completo_clima_elevacion.csv` |
| 1.3 Humedad relativa (ERA5) | `python src/extraer_humedad.py` | `05_dataset_final_para_EDA.csv` |
| 1.4 Vegetación COSCV 2024 | `python src/extraer_vegetacion.py` | `06_dataset_final_enriquecido.csv` |
| 1.5 Pendiente y orientación (SRTM) | `python src/extraer_topografia_final_api.py` | `07_dataset_final_TOPOGRAFICO.csv` |
| 1.6 Pseudo-ausencias — Alternativa A | `python src/generar_ausencias.py` | `08_dataset_modelado_BALANCEADO.csv` |
| 1.7 Pseudo-ausencias — Alternativa B | ver abajo | dataset `_altB` |

```bash
python src/alternativa_b/generar_ausencias_altB.py
python src/alternativa_b/completar_clima_era5_altB.py
python src/alternativa_b/extraer_topografia_final_api_altB.py
```

**Fase 2 · Modelado** — notebooks en `notebooks/`, en orden:

| Notebook | Qué hace |
|---|---|
| `08_Preprocesamiento_Modelado` / `08b_..._altB` | One-hot encoding y partición estratificada 80/20 |
| `09_Entrenamiento_Modelos_Base` | Competición baseline XGBoost vs. LightGBM |
| `10_Optimizacion_Hiperparametros` / `10b_..._altB` | RandomizedSearchCV (50 × 5-fold); **exporta los `.pkl`** a la raíz |
| `11_Calibracion_Explicabilidad_SHAP` / `11b_..._altB` | Barrido de umbrales con F2 y análisis SHAP global y local |

Los notebooks `01`–`07` contienen la limpieza inicial y el análisis exploratorio (temporal, geoespacial y de correlaciones).

**Fase 3 · Despliegue** — con los `.pkl` y las matrices generadas, lanza el dashboard como en el Camino A.

## Estructura del repositorio

```
tfg-incendios-valencia/
├── datos/
│   ├── raw/            # EGIF y COSCV (no versionados)
│   ├── processed/      # Matrices de train/test usadas por el dashboard
│   └── APIs/           # api_AEMET.txt (no versionado)
├── notebooks/          # 01–11: limpieza, EDA, modelado, SHAP
├── src/
│   ├── alternativa_b/  # Pipeline de la Alternativa B
│   ├── run_pipeline.py
│   ├── completar_clima_era5.py
│   ├── extraer_humedad.py
│   ├── extraer_vegetacion.py
│   ├── extraer_topografia_final_api.py
│   ├── generar_ausencias.py
│   └── dashboard.py
├── lgbm_model.pkl
├── lgbm_model_altB.pkl
├── environment.yml
└── requirements.txt
```

## Stack

Python · pandas · GeoPandas · pyproj · scikit-learn · LightGBM · XGBoost · SHAP · Streamlit · Folium · matplotlib · APIs REST (Open-Meteo, Open Topo Data, AEMET OpenData)

## Autor

**Marcos Civera García** — Graduado en Ciencia de Datos (Universitat de València)
Tutor: Antonio Manuel López Quílez

[LinkedIn](https://www.linkedin.com/in/TU-PERFIL) · [Email](mailto:TU-EMAIL)
<!-- Rellena tus enlaces. Si subes la memoria en PDF a docs/, enlázala aquí: [Memoria completa (PDF)](docs/memoria_TFG.pdf) -->
