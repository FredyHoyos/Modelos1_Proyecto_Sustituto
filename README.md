# Proyecto - Modelos 1 (Fase 1)

Este repositorio contiene la **Fase 1** del proyecto de Machine Learning para predecir la duración de viajes de taxi en NYC usando el dataset del reto de Kaggle:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

## Estructura esperada

- `fase-1/notebook.ipynb`
- `data/train.csv`
- `models/model.pkl`

## Requisitos previos: Datos

**Importante:** Los archivos de datos son demasiado grandes para GitHub. Debes descargarlos de Kaggle manualmente:

### Pasos para descargar los datos:

1. Visita el reto en Kaggle:  
   https://www.kaggle.com/competitions/nyc-taxi-trip-duration

2. Aceptar los términos y descargar los archivos:
   - `train.csv` (~191 MB)
   - `test.csv` (~67 MB)

3. Descomprime (si es necesario) y copia los archivos a la carpeta `data/` de este proyecto:
   ```
   data/
   ├── train.csv
   └── test.csv
   ```

**Alternativa con API de Kaggle** (si tienes credenciales configuradas):
```bash
kaggle competitions download -c nyc-taxi-trip-duration -p data/
unzip -d data/ "data/nyc-taxi-trip-duration.zip" 2>/dev/null || true
```

## Requisitos

- Python 3.10 o superior
- Dependencias en `requirements.txt`
- Archivos de datos descargados (ver sección anterior)

## Instalación

1. Abrir una terminal en la raíz del proyecto.
2. (Opcional) Crear entorno virtual.
3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución de la Fase 1

**Prerequisito:** Asegúrate de que `data/train.csv` esté descargado y colocado en la carpeta `data/` (ver sección "Requisitos previos: Datos").

Pasos:

1. Abrir el notebook `fase-1/notebook.ipynb`.
2. Ejecutar todas las celdas en orden (Run All).
3. El notebook realiza los siguientes pasos:
	- Carga `../data/train.csv`
	- Selecciona columnas requeridas y elimina nulos
	- Crea `trip_distance` con fórmula Haversine
	- Define `X` e `y`
	- Divide datos 80/20
	- Entrena `RandomForestRegressor`
	- Evalúa con RMSE
	- Guarda el modelo en `../models/model.pkl`
	- Muestra predicciones de ejemplo

## Resultado esperado

Al finalizar la ejecución del notebook:

- Se imprime el valor de RMSE.
- Se muestra una tabla simple con valores reales y predicciones.
- Se genera o sobrescribe el archivo `models/model.pkl`.

## Nota de alcance

Esta entrega cubre únicamente la **Fase 1**. Docker y etapas posteriores (Fase 2/Fase 3) no forman parte de esta fase.
