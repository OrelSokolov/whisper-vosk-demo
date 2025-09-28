# Используем официальный Python образ
FROM python:3.13-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Создаем директории для аудио файлов и моделей
RUN mkdir -p static/audio models

# Скачиваем и устанавливаем легкую модель Vosk (русская модель)
RUN cd models && \
    wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip && \
    unzip vosk-model-small-ru-0.22.zip && \
    rm vosk-model-small-ru-0.22.zip

# Открываем порт 5000
EXPOSE 5000

# Предварительно загружаем Whisper Turbo модель для ускорения запуска
RUN python3 -c "import whisper; whisper.load_model('turbo')"

# Устанавливаем переменные окружения
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Команда запуска
CMD ["python3", "app.py"]

