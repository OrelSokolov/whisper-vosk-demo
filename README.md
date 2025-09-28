# Speech Recognition Demo Example App

Приложение для распознавания речи с поддержкой Whisper и Vosk.

## Возможности

- 🎤 Запись аудио через браузер
- 🤖 Whisper AI - высокая точность распознавания
- ⚡ Vosk - быстрая обработка
- 🎵 Прослушивание оригинальной записи
- 📝 Автоматическая коррекция грамматики

## Запуск с Docker

### Быстрый старт

```bash
# Сборка и запуск
docker-compose up --build

# Приложение будет доступно на http://localhost:5000
```

### Ручной запуск

```bash
# Сборка образа
docker build -t audio-app .

# Запуск контейнера
docker run -p 5000:5000 \
  -v $(pwd)/static/audio:/app/static/audio \
  audio-app
```

## Запуск без Docker

```bash
# Установка зависимостей
pip install -r requirements.txt

# Скачивание модели Vosk
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip
mv vosk-model-ru-0.42 vosk-model-ru
rm vosk-model-ru-0.42.zip

# Запуск приложения
python3 app.py
```

## Структура проекта

```
audio/
├── app.py                 # Основное приложение Flask
├── templates/
│   └── index.html        # Веб-интерфейс
├── static/
│   └── audio/            # Сохраненные аудио файлы
├── models/
│   └── vosk-model-ru/    # Модель Vosk для русского языка
├── Dockerfile            # Docker конфигурация
├── docker-compose.yml    # Docker Compose конфигурация
└── requirements.txt      # Python зависимости
```

## Использование

1. Откройте http://localhost:5000
2. Выберите движок распознавания (Whisper или Vosk)
3. Нажмите "Начать запись" и говорите
4. Нажмите "Остановить запись"
5. Получите распознанный текст
6. Прослушайте оригинальную запись

## Технические детали

- **Whisper**: OpenAI модель, высокая точность, требует больше ресурсов
- **Vosk**: Открытая модель, быстрая работа, работает офлайн
- **Аудио**: Автоматическая конвертация в формат 16kHz моно для Vosk
- **Грамматика**: Автоматическая коррекция пунктуации и заглавных букв

## Требования

- Docker и Docker Compose (для контейнерного запуска)
- Python 3.13+ (для локального запуска)
- FFmpeg (для обработки аудио)
- 2GB+ RAM (для загрузки моделей)
