FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для MuJoCo + pygame (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости (с кешированием на слое)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . /app/

WORKDIR /app/centipede

# По умолчанию обучение, можно переопределить (docker run centipede play.py --auto)
CMD ["python", "train_overnight.py"]
