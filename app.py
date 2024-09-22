import sys
import os
import cv2
import base64
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
from ultralytics import YOLO

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_count_signal = pyqtSignal(int)

    def __init__(self, video_source=0, model_path='runs/train/yolov10l_trained/weights/best.pt'):
        super().__init__()
        self._run_flag = True
        self.video_source = video_source
        self.model_path = model_path

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

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
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
                # Se for um arquivo de vídeo, reiniciar quando chegar ao fim
                if isinstance(self.video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self._run_flag = False

        # Liberar a captura de vídeo
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detecção de Lixo Aquático em Tempo Real")
        self.setGeometry(100, 100, 800, 600)

        # Criar QLabel para exibir o vídeo
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 700, 500)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px solid black;")

        # Criar QLabel para exibir a contagem de objetos
        self.count_label = QLabel("Objetos Detectados: 0", self)
        self.count_label.setGeometry(50, 560, 300, 30)
        self.count_label.setStyleSheet("font-size: 16px;")

        # Botão para selecionar vídeo
        self.select_button = QPushButton("Selecionar Vídeo", self)
        self.select_button.setGeometry(400, 560, 150, 30)
        self.select_button.clicked.connect(self.select_video)

        # Botão para iniciar a detecção
        self.start_button = QPushButton("Iniciar Detecção", self)
        self.start_button.setGeometry(600, 560, 150, 30)
        self.start_button.clicked.connect(self.start_detection)

        # Variável para armazenar o caminho do vídeo
        self.video_source = 'exemplo.mp4'  # Padrão: vídeo de teste

        # Inicializar o thread de vídeo
        self.thread = None

    def select_video(self):
        # Abrir diálogo para selecionar um arquivo de vídeo
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo de Vídeo", "", "Arquivos de Vídeo (*.mp4 *.avi *.mov)", options=options
        )
        if file_name:
            self.video_source = file_name
            QMessageBox.information(self, "Fonte de Vídeo Selecionada", f"Fonte de vídeo: {self.video_source}")

    def start_detection(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Atenção", "Detecção já está em execução.")
            return

        # Verificar se o arquivo de vídeo existe
        if isinstance(self.video_source, str) and not os.path.exists(self.video_source):
            QMessageBox.critical(self, "Erro", f"Arquivo de vídeo não encontrado: {self.video_source}")
            return

        # Iniciar o thread de vídeo
        self.thread = VideoThread(video_source=self.video_source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_count_signal.connect(self.update_count)
        self.thread.start()

    def closeEvent(self, event):
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
