# TFM — Plataforma E-Commerce IA
**DigitechFP · Especialización en IA y Big Data · 2025-2026**

## Antes de hacer `docker compose up`

### 1. Coloca los datasets de Kaggle en `volumes/hdfs/data/raw/`

Descarga desde Kaggle y copia aquí:

| Archivo | Dataset Kaggle |
|---|---|
| `Reviews.csv` | [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| `train_transaction.csv` | [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) |
| `train_identity.csv` | [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) |

> El File Manager (puerto 3055) también permite subirlos desde el navegador una vez el entorno esté arriba.

### 2. Arranca el entorno

```bash
docker compose up -d
```

La primera vez tarda unos minutos (descarga imágenes + instala librerías extra en Jupyter).

Comprueba que todo está listo:
```bash
docker compose logs -f jupyter
# Espera hasta ver: "Jupyter Server ... is running at:"
```

### 3. Accede a Jupyter Lab

**http://127.0.0.1:3011** — token: `sparklab`

Los notebooks del proyecto están en `work/notebooks/`.

---

## Estructura del proyecto

```
tfm_ecommerce/
├── docker-compose.yml          ← entorno completo
├── requirements.txt            ← dependencias Python
├── info.txt                    ← URLs y credenciales de acceso
├── README.md                   ← este archivo
├── models/
│   ├── models-fraud/           ← lstm_fraud_model.keras + fraud_scaler.joblib
│   ├── models-recommendation/  ← item_similarity_matrix.joblib + kmeans_rfm_model.joblib
│   └── models-sentiment/       ← sentiment_pipeline.pkl
├── reports/
│   ├── reports-dashboard/      ← gráficos consolidados + dashboard_ejecutivo.txt
│   ├── reports-fraud/          ← desbalanceo_clases.png + matriz de confusión
│   ├── reports-recommendation/ ← distribucion_ratings.png + metodo_codo_rfm.png
│   └── reports-sentiment/      ← evaluation_sentiment.png + top_ngrams_sentiment.png
└── volumes/
    ├── hdfs/                   ← datos compartidos entre contenedores
    │   └── data/
    │       ├── raw/            ← ⬅ PON AQUÍ LOS DATASETS DE KAGGLE
    │       ├── clean/          ← salida de los ETLs PySpark
    │       │   ├── ratings.csv
    │       │   ├── sentiment.csv
    │       │   └── fraud_processed.csv
    │       └── features/       ← features engineered
    └── compute-bigdata/
        └── notebooks/          ← notebooks de Jupyter
```

---

## Servicios del clúster

| Servicio | URL | Para qué |
|---|---|---|
| Jupyter Lab | http://127.0.0.1:3011 | Desarrollo de notebooks |
| File Manager | http://127.0.0.1:3055 | Explorar y subir archivos |
| Spark Master UI | http://127.0.0.1:8080 | Estado del clúster |
| Spark Worker 1 | http://127.0.0.1:8081 | Estado del worker |
| Spark Worker 2 | http://127.0.0.1:8082 | Estado del worker |
| Spark App UI | http://localhost:4040 | Monitor de jobs activos |

---

## Notebooks del TFM (orden de ejecución)

### 01 · ETL Amazon Reviews — PySpark
**Entrada:** `data/raw/Reviews.csv` | **Salidas:** `data/clean/ratings.csv`, `data/clean/sentiment.csv`

Pipeline ETL completo sobre el dataset de Amazon Fine Food Reviews ejecutado en el clúster Spark (docker). Incluye:
- Análisis de nulos y limpieza en cascada (nulos en columnas críticas, ratings fuera de rango 1-5, textos < 10 caracteres).
- Normalización de texto: lowercase → strip HTML → eliminar no-letras → colapsar espacios.
- Generación de `sentiment.csv`: etiqueta binaria (0=negativo, 1=positivo), descartando Score=3.
- Generación de `ratings.csv`: filtro de usuarios con ≥ 5 reseñas y productos con ≥ 3 reseñas para garantizar densidad mínima en la matriz usuario-ítem.

### 02 · ETL IEEE-CIS Fraud Detection — PySpark
**Entrada:** `data/raw/train_transaction.csv` + `data/raw/train_identity.csv` | **Salida:** `data/clean/fraud_processed.csv`

Pipeline ETL sobre el dataset de fraude bancario IEEE-CIS ejecutado en el clúster Spark (docker). Incluye:
- LEFT JOIN de las dos fuentes por `TransactionID` (conservar todas las transacciones aunque no tengan datos de identidad).
- Eliminación de columnas con > 50 % de nulos (214 de 434 columnas tras el join).
- Selección manual de features transaccionales relevantes: `TransactionAmt`, `card1-6`, `addr`, features C/D/V seleccionadas y variables categóricas (`ProductCD`, `card4`, `card6`, dominios de email, `DeviceType`).
- Imputación de numéricas con mediana (robusta a distribuciones sesgadas) y codificación de categóricas con `StringIndexer`.
- Ordenación por usuario (`card1`) y tiempo (`TransactionDT`) con `row_number()` para preparar las ventanas deslizantes del LSTM.

### 03 · EDA en RapidMiner
Documentación del análisis exploratorio realizado en RapidMiner. No genera artefactos de datos.

**EDA sobre `ratings.csv`:** K-Means con k=4 sobre dimensiones RFM identificó cuatro segmentos: cluster_0 (~17.000 usuarios ocasionales), cluster_1 (~2.000 usuarios activos —segmento prioritario para recomendación—), cluster_3 (~96 usuarios hiper-activos, posibles revisores profesionales) y cluster_2 (29 usuarios atípicos con timestamps muy antiguos y valoraciones bajas, excluidos del motor).

**EDA sobre `fraud_processed.csv`:** Árbol de decisión (information gain) sobre muestra estratificada de 50.000 transacciones (split 70/30). Accuracy test: 97,01 %. Recall fraude test: 32,50 % — confirma empíricamente la necesidad de SMOTE y una arquitectura LSTM con optimización de F1/Recall en lugar de accuracy.

### 04 · Motor de Recomendación
**Entrada:** `data/clean/ratings.csv` | **Métrica objetivo:** Precision@10 ≥ 0.70 | **Resultado:** 0.7432 ✅

- Segmentación de usuarios con K-Means (k=4) sobre métricas RFM normalizadas con MinMaxScaler. El codo se busca en k∈[2,6] y el modelo se serializa junto al scaler.
- Filtrado colaborativo ítem-ítem: matriz dispersa CSR (ítems × usuarios) y similitud coseno. Las predicciones se normalizan por la suma de similitudes para eliminar sesgo de escala, y los ítems ya consumidos se excluyen forzando su score a −∞.
- Evaluación sobre muestra de 1.000 usuarios: un ítem se considera relevante si el usuario le dio ≥ 4 estrellas.
- Artefactos guardados: `kmeans_rfm_model.joblib`, `rfm_scaler.joblib`, `item_similarity_matrix.joblib`, `item_mapping.joblib`.

### 05 · Detección de Fraude con LSTM
**Entrada:** `data/clean/fraud_processed.csv` | **Métricas objetivo:** F1 ≥ 0.88, Recall ≥ 0.85 | **Resultado:** F1=0.9540, Recall=0.9240 ✅

- Aplicación de SMOTE sobre datos escalados (StandardScaler) para equilibrar el dataset al 50/50 (tasa de fraude original: ~3.5 %).
- Construcción de ventanas deslizantes de 5 pasos temporales → tensor 3D `[muestras, 5, n_features]`.
- Arquitectura LSTM en dos capas (64 → Dropout(0.3) → 32 → Dropout(0.2) → Dense(16, relu) → Dense(1, sigmoid)), compilada con Adam y binary crossentropy.
- Entrenamiento de 30 épocas (batch=512, validation_split=0.1), split estratificado 80/20.
- Artefactos guardados: `lstm_fraud_model.keras`, `fraud_scaler.joblib`.

### 06 · Análisis de Sentimiento
**Entrada:** `data/clean/sentiment.csv` | **Métrica objetivo:** Accuracy ≥ 0.85 | **Resultado:** 0.9519 ✅

- Pipeline TF-IDF + SGDClassifier (loss=`modified_huber`): vocabulario de 150.000 n-gramas (1-3), `sublinear_tf=True`, `class_weight='balanced'` para compensar desbalance 5:1 sin SMOTE.
- Split estratificado 80/20. Se reportan accuracy, F1 por clase y AUC-ROC para detectar predicciones degeneradas sobre la clase mayoritaria.
- Visualizaciones: matriz de confusión, curva ROC, top-20 n-gramas por clase (coeficientes SVM interpretables).
- Función `predecir_sentimiento()` de inferencia con el mismo preprocesado del ETL.
- Artefacto guardado: `sentiment_pipeline.pkl`.

### 07 · Dashboard de KPIs
**Entrada:** resultados de los notebooks 04, 05 y 06 | **Salida:** `reports/reports-dashboard/`

Cuadro de mando ejecutivo que consolida todas las métricas del proyecto. Genera gráficos comparativos (meta vs. real) para los tres módulos, la distribución de segmentos de clientes, la matriz de confusión del detector de fraude, la distribución de sentimientos y el informe final `dashboard_ejecutivo.txt`.

---

## Resumen de métricas finales

| Módulo | Métrica | Objetivo | Resultado |
|---|---|---|---|
| Motor de Recomendación | Precision@10 | ≥ 0.70 | **0.7432** ✅ |
| Detección de Fraude | F1-Score | ≥ 0.88 | **0.9540** ✅ |
| Detección de Fraude | Recall | ≥ 0.85 | **0.9240** ✅ |
| Análisis de Sentimiento | Accuracy | ≥ 0.85 | **0.9519** ✅ |

---

## Conexión a Spark desde cualquier notebook

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TFM_Ecommerce") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.range(5).show()
```

Los datos se leen desde `/opt/spark-data/data/raw/` dentro del contenedor,
que corresponde a `./volumes/hdfs/data/raw/` en tu máquina.

---

## Parar el entorno

```bash
# Parar sin borrar datos
docker compose down

# Parar Y borrar volúmenes de filebrowser (los datos en ./volumes/ se conservan)
docker compose down -v
```