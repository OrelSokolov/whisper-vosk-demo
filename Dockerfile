# Базовый образ с CUDA (runtime + Python)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Чтобы apt не задавал вопросы (tzdata и др.)
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем Python 3.13 и системные зависимости
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
        python3.13 \
        python3.13-venv \
        python3.13-distutils \
        python3-pip \
        ffmpeg \
        wget \
        unzip \
        build-essential \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем локальные колёса
COPY wheels/ /wheels/

# Обновляем pip/setuptools/wheel перед установкой
RUN python3.13 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости только из wheels
RUN python3.13 -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Создаем директории для аудио и моделей
RUN mkdir -p static/audio models

# Скачиваем Vosk модель (если нужно, можно тоже заранее положить в wheels/models)
RUN cd models && \
    wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip && \
    unzip vosk-model-small-ru-0.22.zip && \
    rm vosk-model-small-ru-0.22.zip

# Открываем порт 5000
EXPOSE 5000

# Устанавливаем переменные окружения
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Запуск приложения
CMD ["python3.13", "app.py"]

