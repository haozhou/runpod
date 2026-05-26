FROM runpod/pytorch:1.0.3-dev-fix-image-vulnerabilities-cu1281-torch290-ubuntu2204

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV MODEL_DIR=/runpod-volume/models

CMD ["python", "-u", "handler.py"]
