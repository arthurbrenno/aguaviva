import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from ultralytics import YOLO


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_count_signal = pyqtSignal(int)

    def __init__(self, video_source=0, model_path='runs/train/yolov10l_trained/weights/best.pt'):
        super().__init__()
        self._run_flag = True
        self.video_source = video_source
        self.model_path = model_path
        self.detect = False  # Flag para controlar a detecção

    def run(self):
        # Carregar o modelo YOLO
        if not os.path.exists(self.model_path):
            print(f"Modelo YOLO não encontrado em: {self.model_path}")
            self._run_flag = False
            return

        model = YOLO(self.model_path)

        # Inicializar a captura de vídeo
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Erro ao abrir a fonte de vídeo: {self.video_source}")
            self._run_flag = False
            return

        # Obter FPS do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # FPS padrão caso não seja possível obter
        frame_duration_ms = int(1000 / fps)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                if self.detect:
                    # Realizar a detecção
                    results = model(frame, verbose=False)

                    # Anotar o frame com as detecções
                    annotated_frame = results[0].plot()

                    # Contagem de objetos detectados
                    object_count = len(results[0].boxes)

                    # Emitir sinais para atualizar a interface
                    self.change_pixmap_signal.emit(annotated_frame)
                    self.update_count_signal.emit(object_count)
                else:
                    # Sem detecção, exibir frame original
                    self.change_pixmap_signal.emit(frame)
                    self.update_count_signal.emit(0)

                # Controlar a taxa de quadros
                QThread.msleep(frame_duration_ms)
            else:
                # Se for um arquivo de vídeo, reiniciar quando chegar ao fim
                if isinstance(self.video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self._run_flag = False

        # Liberar a captura de vídeo
        cap.release()

    def stop(self):
        """Pare o thread de vídeo."""
        self._run_flag = False
        self.wait()

    def start_detection(self):
        """Ative a detecção."""
        self.detect = True

    def stop_detection(self):
        """Desative a detecção."""
        self.detect = False


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detecção de Lixo Aquático em Tempo Real")
        self.setGeometry(100, 100, 1200, 800)  # Aumentar o tamanho da janela
        self.setStyleSheet("background-color: #2E3440; color: #D8DEE9;")

        # Inicializar a variável de vídeo antes de usá-la
        self.video_source = 'runs/train/yolov10l_trained/weights/exemplo.mp4'  # Atualize para o caminho correto do seu vídeo de teste

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Cabeçalho
        self.header = QLabel("Detecção de Lixo Aquático em Tempo Real", self)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont('Arial', 24, QFont.Bold))
        self.header.setStyleSheet("color: #81A1C1;")
        self.main_layout.addWidget(self.header)

        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #434C5E;")
        self.main_layout.addWidget(separator)

        # Container para vídeo e contagem
        self.video_container = QVBoxLayout()
        self.main_layout.addLayout(self.video_container)

        # QLabel para exibir o vídeo
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #3B4252; border-radius: 10px;")
        self.label.setFixedSize(1000, 600)  # Aumentar o tamanho do vídeo
        self.video_container.addWidget(self.label)

        # Contagem de objetos detectados
        self.count_label = QLabel("Objetos Detectados: 0", self)
        self.count_label.setFont(QFont('Arial', 18))
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("color: #A3BE8C;")
        self.video_container.addWidget(self.count_label)

        # Botões de controle
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # Botão para selecionar vídeo
        self.select_button = QPushButton("Selecionar Vídeo", self)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #81A1C1;
                color: #2E3440;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.select_button.clicked.connect(self.select_video)
        self.button_layout.addWidget(self.select_button)

        # Botão para usar webcam
        self.webcam_button = QPushButton("Usar Webcam", self)
        self.webcam_button.setStyleSheet("""
            QPushButton {
                background-color: #81A1C1;
                color: #2E3440;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.webcam_button.clicked.connect(self.use_webcam)
        self.button_layout.addWidget(self.webcam_button)

        # Botão para iniciar/parar detecção
        self.start_button = QPushButton("Iniciar Detecção", self)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #88C0D0;
                color: #2E3440;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.start_button.clicked.connect(self.toggle_detection)
        self.button_layout.addWidget(self.start_button)

        # Status de vídeo selecionado
        self.status_label = QLabel(f"Fonte de Vídeo: {self.video_source}", self)
        self.status_label.setFont(QFont('Arial', 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ECEFF4;")
        self.main_layout.addWidget(self.status_label)

        # Inicializar o thread de vídeo
        self.thread = None

    def select_video(self):
        """Abrir diálogo para selecionar um arquivo de vídeo."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo de Vídeo", "", "Arquivos de Vídeo (*.mp4 *.avi *.mov)", options=options
        )
        if file_name:
            self.video_source = file_name
            self.status_label.setText(f"Fonte de Vídeo: {self.video_source}")
            self.display_video()

    def use_webcam(self):
        """Definir a fonte de vídeo para a webcam."""
        self.video_source = 0  # Índice padrão da webcam
        self.status_label.setText("Fonte de Vídeo: Webcam")
        self.display_video()

    def display_video(self):
        """Exibe o vídeo sem detecção."""
        if self.thread:
            self.thread.stop()
            self.thread = None

        # Criar e iniciar o thread de vídeo sem detecção
        self.thread = VideoThread(video_source=self.video_source)
        self.thread.detect = False  # Garantir que a detecção está desativada
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_count_signal.connect(self.update_count)
        self.thread.start()

    def toggle_detection(self):
        """Alterna entre iniciar e parar a detecção."""
        if not self.thread:
            QMessageBox.warning(self, "Atenção", "Selecione uma fonte de vídeo primeiro.", QMessageBox.Ok)
            return

        if not self.thread.detect:
            # Iniciar detecção
            self.thread.start_detection()
            self.start_button.setText("Parar Detecção")
            self.status_label.setText("Detecção em andamento...")
        else:
            # Parar detecção
            self.thread.stop_detection()
            self.start_button.setText("Iniciar Detecção")
            if isinstance(self.video_source, str):
                self.status_label.setText(f"Fonte de Vídeo: {self.video_source}")
            else:
                self.status_label.setText("Fonte de Vídeo: Webcam")

    def closeEvent(self, event):
        """Garantir que o thread de vídeo seja parado ao fechar o aplicativo."""
        if self.thread:
            self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """Atualiza a imagem exibida na interface."""
        qt_img = self.convert_cv_qt(cv_img)
        self.label.setPixmap(qt_img)

    def update_count(self, count):
        """Atualiza a contagem de objetos detectados."""
        self.count_label.setText(f"Objetos Detectados: {count}")

    def convert_cv_qt(self, cv_img):
        """Converte um frame do OpenCV para QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
