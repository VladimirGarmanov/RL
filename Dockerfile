FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libegl1 \
    libgl1 \
    libglfw3 \
    libglew2.2 \
    libglu1-mesa \
    libosmesa6 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

WORKDIR /app/centipede

CMD ["python", "train.py", "--help"]
