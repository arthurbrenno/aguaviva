import os
import cv2
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import eventlet

# Use eventlet para suportar WebSockets
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Carregar o modelo treinado YOLOv10x
MODEL_PATH = os.getenv('MODEL_PATH', 'runs/train/yolov10l_trained/weights/best.pt')
model = YOLO(MODEL_PATH)

def gen_frames():
    """Gera frames processados para transmissão via WebSockets."""
    cap = cv2.VideoCapture(0)  # 0 para webcam; substitua pelo caminho do vídeo se necessário

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Realizar a detecção
            results = model(frame)

            # Anotar o frame com as detecções
            annotated_frame = results[0].plot()

            # Contagem de objetos detectados
            object_count = len(results[0].boxes)

            # Converter o frame para JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Codificar em base64 para transmitir via WebSockets
            encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')

            # Emitir o frame para o frontend
            socketio.emit('video_frame', {'frame': encoded_frame})

            # Emitir a contagem de objetos
            socketio.emit('detection_data', {'count': object_count})

            # Controlar a taxa de transmissão (ajuste conforme necessário)
            socketio.sleep(0.03)  # Aproximadamente 30 FPS

    cap.release()

@app.route('/')
def index():
    """Rota principal que serve a página web."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Evento quando um cliente se conecta."""
    print('Cliente conectado')
    socketio.start_background_task(gen_frames)

@socketio.on('disconnect')
def handle_disconnect():
    """Evento quando um cliente se desconecta."""
    print('Cliente desconectado')

if __name__ == '__main__':
    # Executar o aplicativo com suporte a WebSockets
    socketio.run(app, host='0.0.0.0', port=5000)
