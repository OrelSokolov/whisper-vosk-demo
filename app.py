from flask import Flask, request, jsonify, render_template
import whisper
import tempfile
import os
import re
import time
import shutil
import json
import warnings
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import threading

# Отключаем предупреждения Whisper о FP16
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Импорты для Vosk
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Классы для работы с разными бэкендами
class WhisperBackend:
    def __init__(self):
        self.model = None
        self.name = "Whisper"
        self.loading = False
        self.loaded = False
        self.error = None
    
    def load_model_async(self):
        """Асинхронная загрузка модели"""
        if self.loading or self.loaded:
            return
        
        self.loading = True
        self.error = None
        
        def load():
            try:
                print("🔄 Загружаю Whisper Medium по требованию...")
                self.model = whisper.load_model("medium")
                self.loaded = True
                print("✅ Whisper Medium загружен!")
            except Exception as e:
                self.error = str(e)
                print(f"❌ Ошибка загрузки Whisper: {e}")
            finally:
                self.loading = False
        
        # Запускаем загрузку в отдельном потоке
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()
    
    def is_ready(self):
        """Проверяет, готова ли модель к использованию"""
        return self.loaded and self.model is not None
    
    def is_loading(self):
        """Проверяет, загружается ли модель"""
        return self.loading
    
    def get_error(self):
        """Возвращает ошибку загрузки, если есть"""
        return self.error
    
    def transcribe(self, audio_file_path):
        if not self.is_ready():
            if self.is_loading():
                return "Whisper загружается, пожалуйста, подождите..."
            elif self.get_error():
                return f"Ошибка Whisper: {self.get_error()}"
            else:
                return "Whisper не готов, попробуйте позже"
        
        try:
            result = self.model.transcribe(
                audio_file_path,
                language="ru",
                task="transcribe",
                temperature=0.0,
                best_of=5,
                beam_size=5,
                patience=1.0,
                length_penalty=1.0,
                suppress_tokens=[-1],
                initial_prompt="Это аудиозапись на русском языке. Текст должен быть грамматически правильным и читаемым."
            )
            return result['text'].strip()
        except Exception as e:
            return f"Ошибка распознавания Whisper: {str(e)}"

class VoskBackend:
    def __init__(self):
        self.model = None
        self.name = "Vosk"
        self.load_model()
    
    def load_model(self):
        if not VOSK_AVAILABLE:
            return
        
        # Путь к модели Vosk
        model_path = os.path.join(os.getcwd(), "models", "vosk-model-small-ru-0.22")
        if not os.path.exists(model_path):
            # Создаем папку для моделей
            os.makedirs("models", exist_ok=True)
            print("Модель Vosk не найдена. Скачайте её с https://alphacephei.com/vosk/models")
            return
        
        try:
            print(f"🔄 Загружаю модель Vosk из: {model_path}")
            self.model = vosk.Model(model_path)
            print(f"✅ Модель Vosk успешно загружена!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели Vosk: {e}")
    
    def transcribe(self, audio_file_path):
        if not self.model:
            return "Модель Vosk не загружена"
        
        try:
            import wave
            import subprocess
            import tempfile
            
            # Конвертируем аудио в правильный формат для Vosk
            # Vosk требует 16kHz моно WAV файл
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Используем ffmpeg для конвертации с правильными параметрами
            cmd = [
                'ffmpeg', '-i', audio_file_path,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # mono
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-f', 'wav',     # WAV format
                '-y',            # overwrite
                temp_wav.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return f"Ошибка конвертации аудио: {result.stderr}"
            
            print(f"FFmpeg успешно сконвертировал: {audio_file_path} -> {temp_wav.name}")
            
            # Теперь распознаем сконвертированный файл
            wf = wave.open(temp_wav.name, 'rb')
            print(f"Аудио параметры: channels={wf.getnchannels()}, sample_width={wf.getsampwidth()}, framerate={wf.getframerate()}, frames={wf.getnframes()}")
            
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                wf.close()
                os.unlink(temp_wav.name)
                return f"Неверный формат аудио: channels={wf.getnchannels()}, width={wf.getsampwidth()}, rate={wf.getframerate()}"
            
            # Проверяем, есть ли аудио данные
            if wf.getnframes() == 0:
                wf.close()
                os.unlink(temp_wav.name)
                return "Аудио файл пуст"
            
            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            print(f"Vosk recognizer создан для {wf.getframerate()}Hz")
            
            result_text = ""
            chunk_count = 0
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                chunk_count += 1
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    partial_text = result.get('text', '')
                    if partial_text:
                        print(f"Chunk {chunk_count}: {partial_text}")
                        result_text += partial_text + ' '
            
            # Финальный результат
            result = json.loads(rec.FinalResult())
            final_text = result.get('text', '')
            if final_text:
                print(f"Final result: {final_text}")
                result_text += final_text
            
            print(f"Общий результат Vosk: '{result_text.strip()}'")
            
            # Сохраняем длину аудио перед закрытием
            audio_duration = wf.getnframes() / wf.getframerate()
            wf.close()
            os.unlink(temp_wav.name)
            
            # Если ничего не распознано, возвращаем специальное сообщение
            if not result_text.strip():
                return f"Vosk не смог распознать речь. Длина аудио: {audio_duration:.1f} сек. Проверьте громкость и четкость речи."
            
            return result_text.strip()
        except Exception as e:
            return f"Ошибка распознавания Vosk: {str(e)}"

# Инициализируем бэкенды
whisper_backend = None
vosk_backend = None

# Текущий активный бэкенд
current_backend = None

def get_whisper_backend():
    global whisper_backend
    return whisper_backend

def get_vosk_backend():
    global vosk_backend
    if vosk_backend is None and VOSK_AVAILABLE:
        print("🔄 Загружаю Vosk по требованию...")
        vosk_backend = VoskBackend()
        print("✅ Vosk загружен!")
    return vosk_backend

# Инициализируем бэкенды сразу при старте
print("🚀 Запуск приложения...")

# Загружаем Vosk
if VOSK_AVAILABLE:
    print("🔄 Загружаю Vosk при старте...")
    vosk_backend = VoskBackend()
    print("✅ Vosk загружен!")

# Загружаем Whisper синхронно при старте
print("🔄 Загружаю Whisper Turbo при старте...")
whisper_backend = WhisperBackend()
try:
    whisper_backend.model = whisper.load_model("turbo")
    whisper_backend.loaded = True
    print("✅ Whisper Turbo загружен!")
except Exception as e:
    whisper_backend.error = str(e)
    print(f"❌ Ошибка загрузки Whisper: {e}")

# Устанавливаем Vosk как основной бэкенд
if VOSK_AVAILABLE and vosk_backend:
    current_backend = vosk_backend
    print("✅ Vosk установлен как основной бэкенд!")
else:
    current_backend = whisper_backend
    print("✅ Whisper установлен как основной бэкенд!")

def post_process_text(text):
    """
    Универсальная постобработка текста для улучшения читаемости
    """
    # Исправляем пунктуацию
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Убираем пробелы перед знаками
    text = re.sub(r'([.!?])\s*([а-яё])', r'\1 \2', text)  # Добавляем пробелы после точек
    
    # Исправляем заглавные буквы в начале предложений
    sentences = re.split(r'([.!?])', text)
    result = []
    for i, sentence in enumerate(sentences):
        if sentence.strip() and not sentence in '.!?':
            # Делаем первую букву заглавной
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            result.append(sentence)
        else:
            result.append(sentence)
    
    text = ''.join(result)
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/backend', methods=['POST'])
def switch_backend():
    global current_backend
    data = request.get_json()
    backend_name = data.get('backend', 'whisper')
    
    if backend_name == 'vosk':
        vosk_backend_instance = get_vosk_backend()
        if vosk_backend_instance:
            current_backend = vosk_backend_instance
            return jsonify({'success': True, 'backend': 'vosk', 'message': 'Переключено на Vosk'})
        else:
            return jsonify({'success': False, 'error': 'Vosk недоступен'})
    elif backend_name == 'whisper':
        whisper_backend_instance = get_whisper_backend()
        current_backend = whisper_backend_instance
        
        if whisper_backend_instance.is_ready():
            return jsonify({'success': True, 'backend': 'whisper', 'message': 'Переключено на Whisper Turbo'})
        else:
            return jsonify({'success': False, 'backend': 'whisper', 'error': f'Whisper недоступен: {whisper_backend_instance.get_error()}'})
    else:
        return jsonify({'success': False, 'error': 'Неизвестный бэкенд'})

@app.route('/api/whisper/status')
def get_whisper_status():
    """Проверяет статус загрузки Whisper"""
    whisper_backend_instance = get_whisper_backend()
    
    if whisper_backend_instance.is_ready():
        return jsonify({'status': 'ready', 'message': 'Whisper готов к использованию'})
    elif whisper_backend_instance.is_loading():
        return jsonify({'status': 'loading', 'message': 'Whisper загружается...'})
    elif whisper_backend_instance.get_error():
        return jsonify({'status': 'error', 'error': whisper_backend_instance.get_error()})
    else:
        return jsonify({'status': 'not_started', 'message': 'Whisper не запущен'})

@app.route('/api/backends')
def get_backends():
    backends = [{'name': 'whisper', 'available': True, 'description': 'OpenAI Whisper Turbo - быстрая обработка'}]
    
    # Проверяем доступность Vosk
    if VOSK_AVAILABLE:
        backends.append({'name': 'vosk', 'available': True, 'description': 'Vosk - быстрая обработка'})
    else:
        backends.append({'name': 'vosk', 'available': False, 'description': 'Vosk - недоступен'})
    
    return jsonify({
        'current': current_backend.name.lower(),
        'backends': backends
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            
            # Транскрибируем аудио с помощью текущего бэкенда
            original_text = current_backend.transcribe(tmp_file.name)
            
            # Постобработка текста для улучшения грамматики
            text = post_process_text(original_text)
            
            # Создаем постоянный файл для прослушивания
            audio_filename = f"audio_{int(time.time())}.wav"
            audio_path = os.path.join("static", "audio", audio_filename)
            
            # Создаем папку если её нет
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Копируем файл в статическую папку
            shutil.copy2(tmp_file.name, audio_path)
            
            # Удаляем временный файл
            os.unlink(tmp_file.name)
            
            return jsonify({
                'success': True,
                'text': text,
                'original': original_text,  # Оригинальный текст для сравнения
                'audio_url': f"/static/audio/{audio_filename}",  # URL для прослушивания
                'backend': current_backend.name  # Информация о бэкенде
            })
            
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
