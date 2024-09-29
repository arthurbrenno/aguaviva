import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QComboBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize

from ultralytics import YOLO


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_count_signal = pyqtSignal(int)

    def __init__(self, video_source=0, model_path='runs/train/large_finetuned/weights/best.pt'):
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
        self.setWindowTitle("🦈 Detecção de Lixo Aquático em Tempo Real 🐢")
        self.setGeometry(100, 100, 1200, 700)  # Tamanho inicial da janela
        self.setFixedSize(1200, 700)  # Fixar o tamanho da janela para evitar crescimento

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E2A38;
                color: #D8DEE9;
            }
            QLabel {
                color: #ECEFF4;
            }
            QComboBox {
                background-color: #3B4252;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton {
                border: none;
            }
        """)

        # Construir caminho absoluto para o vídeo padrão
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_video_path = os.path.join(script_dir, 'videos', 'exemplo.mp4')
        self.video_source = default_video_path  # Caminho absoluto para o vídeo de teste

        # Construir caminho absoluto para o modelo padrão
        self.model_path = os.path.join(script_dir, 'runs', 'train', 'large_finetuned', 'weights', 'best.pt')  # Modelo padrão

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout principal horizontal
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Seção de Vídeo e Métricas
        self.video_metrics_layout = QVBoxLayout()
        self.main_layout.addLayout(self.video_metrics_layout, stretch=8)

        # QLabel para exibir o vídeo
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: #3B4252;
            border-radius: 15px;
            border: 2px solid #81A1C1;
        """)
        self.label.setFixedSize(800, 500)  # Definir tamanho fixo para o vídeo
        self.video_metrics_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        # Contagem de objetos detectados
        self.count_label = QLabel("🔍 Objetos Detectados: 0", self)
        self.count_label.setFont(QFont('Arial', 18))
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("color: #A3BE8C;")
        self.video_metrics_layout.addWidget(self.count_label)

        # Seleção de Modelo
        self.model_selection_layout = QHBoxLayout()
        self.video_metrics_layout.addLayout(self.model_selection_layout)

        self.model_label = QLabel("🛠️ Modelo:")
        self.model_label.setFont(QFont('Arial', 14))
        self.model_selection_layout.addWidget(self.model_label)

        self.model_combo = QComboBox(self)
        self.model_combo.setFixedWidth(200)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #3B4252;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        self.populate_models()
        self.model_combo.currentIndexChanged.connect(self.change_model)
        self.model_selection_layout.addWidget(self.model_combo)

        # Espaçador para alinhar à esquerda
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.model_selection_layout.addItem(spacer)

        # Status do Modelo e Fonte de Vídeo
        self.status_label = QLabel(f"📡 Fonte de Vídeo: Nenhuma selecionada\n🛠️ Modelo: {self.get_current_model_name()}", self)
        self.status_label.setFont(QFont('Arial', 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ECEFF4;")
        self.video_metrics_layout.addWidget(self.status_label)

        # Seção de Botões (Barra Lateral)
        self.button_layout = QVBoxLayout()
        self.main_layout.addLayout(self.button_layout, stretch=2)

        # Logotipo e Título na Barra Lateral
        self.logo_label = QLabel(self)
        logo_path = os.path.join(script_dir, 'assets', 'logo.png')  # Caminho absoluto para o seu logotipo
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = self.get_circular_pixmap(logo_pixmap, QSize(80, 80))
            self.logo_label.setPixmap(logo_pixmap)
        else:
            self.logo_label.setText("🛠️")  # Emoji substituto caso o logotipo não seja encontrado
            self.logo_label.setFont(QFont('Arial', 24))
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.button_layout.addWidget(self.logo_label)

        self.header_title = QLabel("🦈 Projeto Água Viva 🐢")
        self.header_title.setFont(QFont('Arial', 16, QFont.Bold))
        self.header_title.setAlignment(Qt.AlignCenter)
        self.button_layout.addWidget(self.header_title)

        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #434C5E;")
        self.button_layout.addWidget(separator)

        # Botões de controle
        self.button_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Botão para selecionar vídeo
        self.select_button = QPushButton("📂 Selecionar Vídeo", self)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #81A1C1;
                color: #2E3440;
                padding: 10px 20px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.select_button.clicked.connect(self.select_video)
        self.button_layout.addWidget(self.select_button)

        # Botão para usar webcam
        self.webcam_button = QPushButton("📷 Usar Webcam", self)
        self.webcam_button.setStyleSheet("""
            QPushButton {
                background-color: #81A1C1;
                color: #2E3440;
                padding: 10px 20px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.webcam_button.clicked.connect(self.use_webcam)
        self.button_layout.addWidget(self.webcam_button)

        # Botão para iniciar/parar detecção
        self.start_button = QPushButton("🚀 Iniciar Detecção", self)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #88C0D0;
                color: #2E3440;
                padding: 10px 20px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)
        self.start_button.clicked.connect(self.toggle_detection)
        self.button_layout.addWidget(self.start_button)

        # Espaçador no final da barra lateral
        self.button_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Inicializar o thread de vídeo como None
        self.thread = None

        # Iniciar a reprodução do vídeo padrão automaticamente
        self.display_video()

    def get_circular_pixmap(self, pixmap, size):
        """Retorna um QPixmap circular."""
        # Escalar o pixmap para o tamanho desejado
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        # Criar uma máscara circular
        mask = QPixmap(size)
        mask.fill(Qt.transparent)

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)
        painter.drawEllipse(0, 0, size.width(), size.height())
        painter.end()

        # Aplicar a máscara ao pixmap
        pixmap.setMask(mask.createMaskFromColor(Qt.transparent, Qt.MaskInColor))
        return pixmap

    def populate_models(self):
        """Popula o QComboBox com os modelos disponíveis."""
        self.model_combo.clear()
        train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'train')
        if not os.path.exists(train_dir):
            QMessageBox.warning(self, "⚠️ Atenção", f"Diretório {train_dir} não encontrado.", QMessageBox.Ok)
            return

        models = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if not models:
            QMessageBox.warning(self, "⚠️ Atenção", "Nenhum modelo encontrado em runs/train/.", QMessageBox.Ok)
            return

        self.model_combo.addItems(models)

    def get_current_model_name(self):
        """Retorna o nome do modelo atual."""
        current_model = self.model_combo.currentText()
        return current_model if current_model else "modelo padrão"

    def change_model(self, index):
        """Atualiza o modelo YOLO com base na seleção do usuário."""
        selected_model = self.model_combo.currentText()
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'train', selected_model, 'weights', 'best.pt')
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "⚠️ Atenção", f"Modelo selecionado não encontrado: {self.model_path}", QMessageBox.Ok)
            return
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.display_video()  # Reiniciar a captura com o novo modelo
        self.status_label.setText(f"📡 Fonte de Vídeo: {self.video_source if self.video_source else 'Nenhuma selecionada'}\n🛠️ Modelo: {selected_model}")

    def select_video(self):
        """Abrir diálogo para selecionar um arquivo de vídeo."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo de Vídeo", "", "Arquivos de Vídeo (*.mp4 *.avi *.mov)", options=options
        )
        if file_name:
            self.video_source = file_name
            self.status_label.setText(f"📡 Fonte de Vídeo: {self.video_source}\n🛠️ Modelo: {self.get_current_model_name()}")
            self.display_video()

    def use_webcam(self):
        """Definir a fonte de vídeo para a webcam."""
        self.video_source = 0  # Índice padrão da webcam
        self.status_label.setText(f"📡 Fonte de Vídeo: Webcam\n🛠️ Modelo: {self.get_current_model_name()}")
        self.display_video()

    def display_video(self):
        """Exibe o vídeo sem detecção."""
        if self.thread:
            self.thread.stop()
            self.thread = None

        # Verificar se a fonte de vídeo é válida
        if isinstance(self.video_source, str) and not os.path.exists(self.video_source):
            QMessageBox.warning(self, "⚠️ Atenção", f"Fonte de vídeo não encontrada: {self.video_source}", QMessageBox.Ok)
            return

        # Criar e iniciar o thread de vídeo sem detecção
        self.thread = VideoThread(video_source=self.video_source, model_path=self.model_path)
        self.thread.detect = False  # Garantir que a detecção está desativada
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_count_signal.connect(self.update_count)
        self.thread.start()

        # Atualizar o status para refletir a fonte de vídeo atual
        fonte = "Webcam" if self.video_source == 0 else self.video_source
        modelo = self.get_current_model_name()
        self.status_label.setText(f"📡 Fonte de Vídeo: {fonte}\n🛠️ Modelo: {modelo}")

    def toggle_detection(self):
        """Alterna entre iniciar e parar a detecção."""
        if not self.thread:
            QMessageBox.warning(self, "⚠️ Atenção", "Selecione uma fonte de vídeo primeiro.", QMessageBox.Ok)
            return

        if not self.thread.detect:
            # Iniciar detecção
            self.thread.start_detection()
            self.start_button.setText("🛑 Parar Detecção")
            self.status_label.setText(f"🔄 Detecção em andamento...\n📡 Fonte de Vídeo: {self.video_source if self.video_source else 'Nenhuma selecionada'}\n🛠️ Modelo: {self.get_current_model_name()}")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #BF616A;
                    color: #2E3440;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #D08770;
                }
            """)
        else:
            # Parar detecção
            self.thread.stop_detection()
            self.start_button.setText("🚀 Iniciar Detecção")
            fonte = "Webcam" if self.video_source == 0 else self.video_source
            self.status_label.setText(f"📡 Fonte de Vídeo: {fonte}\n🛠️ Modelo: {self.get_current_model_name()}")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #88C0D0;
                    color: #2E3440;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5E81AC;
                }
            """)

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
        self.count_label.setText(f"🔍 Objetos Detectados: {count}")

    def convert_cv_qt(self, cv_img):
        """Converte um frame do OpenCV para QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())