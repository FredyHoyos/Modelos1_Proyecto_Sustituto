FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fase-2 /app/fase-2
COPY data /app/data
COPY models /app/models

RUN mkdir -p /app/predictions /app/output

VOLUME ["/app/predictions", "/app/models"]

CMD ["bash"]