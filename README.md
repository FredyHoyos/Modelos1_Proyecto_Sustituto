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






# Fase 2

En la Fase 2 se añade un contenedor Docker con los scripts `train.py` y `predict.py` para entrenar e inferir usando el modelo guardado en disco. Esta etapa no modifica la Fase 1; solo reutiliza el mismo contrato de datos y el mismo archivo `models/model.pkl`.

### Estructura de la Fase 2

- `fase-2/train.py`: entrena de nuevo el modelo a partir de un CSV con etiquetas y guarda el resultado en disco.
- `fase-2/predict.py`: carga un modelo previamente entrenado y escribe un archivo CSV con las predicciones.
- `fase-2/Dockerfile`: construye la imagen de Docker con Python y las dependencias necesarias.

### Requisitos de la Fase 2

Para que el contenedor funcione correctamente, `requirements.txt` debe incluir al menos estas librerías:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

### Cómo correr la Fase 2

#### Necesario

1. Verifica que los archivos de datos existan en la carpeta `data/`.
	- Debe existir `data/train.csv` para entrenamiento.
	- Debe existir `data/test.csv` para predicción.
2. Construye la imagen Docker (Dockerfile en `fase-2/`).

```bash
# Desde la raíz del proyecto, especificando el Dockerfile de fase-2
docker build -t taxi-ml -f fase-2/Dockerfile .
```

3. Ejecuta los scripts de entrenamiento e inferencia.

#### Opcional

### Opción A — Interactivo (depuración e inspección)

Si necesitas inspeccionar archivos o ejecutar comandos paso a paso, usa un contenedor con shell y monta volúmenes para acceder a `data`, `models` y `predictions` desde tu máquina:

- Ejecutar un contenedor interactivo (shell):

```bash
docker run --rm -it taxi-ml
```


```bash
docker run --rm -it -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" -v "%cd%/predictions:/app/predictions" taxi-ml bash
```

Dentro del contenedor puedes ejecutar los scripts manualmente, por ejemplo:

```bash
python fase-2/train.py --input_file data/train.csv
```

También puedes ejecutar inferencia desde el contenedor interactivo:

```bash
python fase-2/predict.py --input_file data/test.csv --output_file predictions/output.csv
```

Cuando termines, sal del contenedor escribiendo `exit` o presionando Ctrl+D (esto detiene el contenedor).

---

### Opción B — No interactivo (recomendado para ejecuciones reproducibles)

Para ejecutar los scripts directamente sin abrir un shell dentro del contenedor:

```bash
docker run --rm taxi-ml python fase-2/train.py --input_file data/train.csv
```

- Ejecutar inferencia (genera `predictions/output.csv` dentro del contenedor):

```bash
docker run --rm taxi-ml python fase-2/predict.py --input_file data/test.csv
```

4. Guardar resultados en tu máquina.

	Esto es opcional si solo quieres ejecutar dentro del contenedor, pero es lo recomendable si necesitas conservar `models/` y `predictions/` en tu PC.

Si quieres que `models/` y `predictions/` se sincronicen con tu PC, monta volúmenes:

```bash
# Entrenar (guarda modelo en ./models)
docker run --rm -v "%cd%/models:/app/models" -v "%cd%/predictions:/app/predictions" taxi-ml python fase-2/train.py --input_file data/train.csv

# Predecir (guarda predicciones en ./predictions)
docker run --rm -v "%cd%/models:/app/models" -v "%cd%/predictions:/app/predictions" taxi-ml python fase-2/predict.py --input_file data/test.csv --output_file predictions/output.csv
```

> Nota: En PowerShell usa `%cd%` como en los ejemplos anteriores; en Linux/macOS usa `${PWD}`.

5. Ejecutar desde Docker Desktop UI (alternativa):

	También es opcional; sirve si prefieres usar la interfaz gráfica en lugar de la terminal.

- Abre Docker Desktop → pestaña `Images` → selecciona `taxi-ml` → Run.
- En la ventana de Run añade dos mounts (volúmenes) para poder ver `models` y `predictions` en tu PC:
	- Host path: `<tu_repo>/models` → Container: `/app/models`
	- Host path: `<tu_repo>/predictions` → Container: `/app/predictions`
- Ejecuta el comando de Entrenamiento o Predicción como `Command` o abre un shell y corre los scripts.

6. Rutas internas recomendadas dentro del contenedor

- `/app/data/train.csv`  (datos de entrenamiento)
- `/app/data/test.csv`   (datos de inferencia)
- `/app/models/model.pkl` (modelo serializado)
- `/app/predictions/output.csv` (archivo CSV de salida)

7. Requisitos y notas

- Asegúrate de tener `data/train.csv` y `data/test.csv` descargados en `data/`.
- Instala dependencias locales con `pip install -r requirements.txt` si quieres ejecutar los scripts fuera del contenedor.
- El `Dockerfile` de Fase 2 ya crea `/app/predictions` y expone `/app/models` y `/app/predictions` como volúmenes.

8. Ejemplos rápidos (PowerShell)

```powershell
# Construir
docker build -t taxi-ml -f fase-2/Dockerfile .

# Entrenar y volcar el modelo en ./models
docker run --rm -v "%cd%/models:/app/models" -v "%cd%/predictions:/app/predictions" taxi-ml python fase-2/train.py --input_file data/train.csv

# Predecir y guardar CSV en ./predictions
docker run --rm -v "%cd%/models:/app/models" -v "%cd%/predictions:/app/predictions" taxi-ml python fase-2/predict.py --input_file data/test.csv --output_file predictions/output.csv
```

7. Revisa el archivo generado para confirmar que la predicción quedó guardada en CSV y que la columna principal se llama `trip_duration`.

### Rutas dentro de Docker

Dentro del contenedor no conviene usar rutas como `../`. El proyecto trabaja desde `/app`, así que las rutas internas recomendadas son:

- `/app/data/train.csv`
- `/app/data/test.csv`
- `/app/models/model.pkl`
- `/app/predictions/output.csv`

### Detalles de ejecución

El comando por defecto del Dockerfile es `bash`, para que el contenedor sea flexible y puedas ejecutar manualmente `train.py` o `predict.py` con los argumentos que necesites. Si prefieres automatizar el entrenamiento por defecto, puedes cambiar ese `CMD`, pero para entrega utilizare mejor `bash`.

### Notas de implementación

- `train.py` sobrescribe `models/model.pkl` cuando se reentrena el modelo.
- `predict.py` escribe un CSV reproducible, no solo imprime resultados por consola.
- El preprocesado de Fase 2 replica la lógica de la Fase 1 para evitar diferencias entre entrenamiento e inferencia.
- Las predicciones se generan con el mismo orden de filas que sobreviven al preprocesado, por eso el índice se mantiene alineado antes de escribir el archivo final.




Modelos1_Proyecto_sustituto — Estructura del repositorio

| Ruta | Tipo | Descripción |
|---|---:|---|
| README.md | Archivo | Instrucciones y notas del proyecto |
| requirements.txt | Archivo | Dependencias Python |
| data/ | Carpeta | Datos (no incluidos en repo) |
| data/train.csv | Archivo | CSV de entrenamiento (local) |
| data/test.csv | Archivo | CSV de inferencia (local) |
| models/ | Carpeta | Modelos serializados |
| models/model.pkl | Archivo | Modelo entrenado (pickle) |
| fase-1/ | Carpeta | Notebooks y experimentos (Fase 1) |
| fase-1/notebook.ipynb | Notebook | Cuaderno de análisis y entrenamiento |
| fase-2/ | Carpeta | Scripts y Docker para entrenamiento/inf.
| fase-2/train.py | Script | Script de entrenamiento (CLI) |
| fase-2/predict.py | Script | Script de inferencia (CLI) |
| fase-2/Dockerfile | Archivo | Dockerfile para Fase 2 |
| fase-3/ (próximamente) | Carpeta | Extensiones futuras (API, cliente) |
| fase-3/train.py | Script | (Próximo) entrenamiento de fase 3 |
| fase-3/predict.py | Script | (Próximo) inferencia fase 3 |
| fase-3/apirest.py | Script | (Próximo) servidor API REST |
| fase-3/client.py | Script | (Próximo) cliente de ejemplo |
| fase-3/Dockerfile | Archivo | (Próximo) Dockerfile para Fase 3 |