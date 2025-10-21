# Dockerfile.train
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# evita output buffering di Python
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copia codice
COPY . /workspace

# Comando di default (esempio: avvia il training)
CMD ["python", "training/train_eng_BPE.py"]