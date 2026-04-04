# Proyecto - Modelos 1 (Fase 1)

Este repositorio contiene la **Fase 1** del proyecto de Machine Learning para predecir la duración de viajes de taxi en NYC usando el dataset del reto de Kaggle:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

## Estructura esperada

- `fase-1/notebook.ipynb`
- `data/train.csv`
- `models/model.pkl`

## Requisitos previos: Datos

**Importante:** Los archivos de datos son demasiado grandes para GitHub. Debes descargarlos de Kaggle manualmente:

## Antes de empezar

1. Clona el repositorio:
	```bash
	git clone https://github.com/FredyHoyos/Modelos1_Proyecto_Sustituto.git
	```
2. Entra a la carpeta del proyecto:
	```bash
	cd Modelos1_Proyecto_Sustituto
	```
3. Si todavía no tienes acceso a Kaggle, crea tu cuenta y acepta las reglas de la competencia antes de descargar los datos.

### Pasos para descargar los datos:

1. Visita el reto en Kaggle:  
   https://www.kaggle.com/competitions/nyc-taxi-trip-duration

2. Aceptar los términos y descargar los archivos:
   - `train.csv` (~196 MB)
   - `test.csv` (~69 MB)

3. Descomprime (si es necesario) y copia los archivos a la carpeta `data/` de este proyecto:
   ```
   data/
   ├── train.csv
   └── test.csv
   ```

	La carpeta `data/` ya está incluida en el repositorio con un archivo marcador, pero los CSV no se suben a GitHub por tamaño. Debes descargarlos tú y colocarlos ahí localmente.

**Alternativa con API de Kaggle** (si tienes credenciales configuradas):
```bash
kaggle competitions download -c nyc-taxi-trip-duration -p data/
```

Si el archivo descargado viene comprimido, descomprímelo manualmente y deja `train.csv` y `test.csv` dentro de `data/`.

## Requisitos

- Python 3.10 o superior
- Dependencias en `requirements.txt`
- Archivos de datos descargados (ver sección anterior)

## Instalación

1. Abrir una terminal en la raíz del proyecto.
2. (Opcional) Crear y activar un entorno virtual.
3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Si vas a usar el comando de Kaggle, instala y configura la CLI de Kaggle con tus credenciales.

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

4. Verificar que el archivo `models/model.pkl` se haya generado o actualizado al final de la ejecución.

## Resultado esperado

Al finalizar la ejecución del notebook:

- Se imprime el valor de RMSE.
- Se muestra una tabla simple con valores reales y predicciones.
- Se genera o sobrescribe el archivo `models/model.pkl`.

## Nota de alcance

Esta entrega cubre únicamente la **Fase 1**. Docker y etapas posteriores (Fase 2/Fase 3) no forman parte de esta fase.
