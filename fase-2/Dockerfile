FROM python:3.10-slim
# Usa una imagen base de Python ligera y estable (slim).

WORKDIR /app
# Establece el directorio de trabajo en /app para mantener rutas relativas.

COPY requirements.txt /app/requirements.txt
# Copia el fichero de dependencias al contenedor para instalar paquetes.

RUN pip install --no-cache-dir -r /app/requirements.txt
# Instala las dependencias listadas usando pip sin cache para reducir la imagen.

COPY fase-2 /app/fase-2
# Copia los scripts y módulos de la Fase 2 al contenedor.

COPY data /app/data
# Copia los datos de ejemplo/local (el evaluador montará volúmenes cuando sea necesario).

COPY models /app/models
# Copia el directorio models para incluir un modelo preentrenado si existe.

RUN mkdir -p /app/predictions /app/output
# Crea las carpetas de salida que usarán `train.py` y `predict.py`.

VOLUME ["/app/predictions", "/app/models"]
# Declara volúmenes para permitir mount desde el host (persistencia de modelos y predicciones).

CMD ["bash"]
# Entrada por defecto: arranca una shell para permitir ejecuciones interactivas o comandos.