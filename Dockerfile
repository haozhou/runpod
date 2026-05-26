FROM runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV MODEL_DIR=/runpod-volume/models

CMD ["python", "-u", "handler.py"]
