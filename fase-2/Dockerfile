# Imagen base ligera para el entorno de ejecución de ML.
FROM python:3.10-slim

# Trabajamos dentro de /app para que las rutas relativas se mantengan estables.
WORKDIR /app

# Instala dependencias antes de copiar el código para aprovechar la caché de Docker.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia el código fuente y los conjuntos de datos locales que esperan los scripts.
COPY fase-2 /app/fase-2
COPY data /app/data
COPY models /app/models

# Crea los directorios de salida usados por entrenamiento y predicción.
RUN mkdir -p /app/predictions /app/output

# Expone las carpetas de modelos y predicciones como puntos de montaje.
VOLUME ["/app/predictions", "/app/models"]

# Deja el contenedor interactivo por defecto para poder ejecutar comandos manualmente.
CMD ["bash"]