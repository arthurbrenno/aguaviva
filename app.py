import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time

# Configuração da página
st.set_page_config(page_title="Água Viva", page_icon="🌊", layout="wide")

# Carregar e exibir o logo
logo_path = os.path.join(os.path.dirname(__file__), 'assets/logo.png')
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning(f"Logo não encontrado no caminho: {logo_path}")

st.title("🌊 Água Viva - Detecção de Lixo Subaquático")

# Painel lateral de configurações
st.sidebar.title("Configurações")

# Seleção do modelo
weights_dir = 'weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
if model_files:
    selected_model = st.sidebar.selectbox("Selecione o modelo", model_files)

    # Carregar o modelo selecionado
    @st.cache_resource
    def load_model(model_path):
        return YOLO(model_path)

    model_path = os.path.join(weights_dir, selected_model)
    model = load_model(model_path)

    # Filtragem de classes
    all_classes = list(model.names.values())
    selected_classes = st.sidebar.multiselect("Selecione as classes para exibir", all_classes, default=all_classes)

    # Configurações adicionais
    st.sidebar.subheader("Ajustes Adicionais")
    confidence_threshold = st.sidebar.slider("Limite de Confiança", 0.0, 1.0, 0.25)
    display_fps = st.sidebar.checkbox("Exibir FPS", value=True)

    # Seleção da fonte de vídeo
    video_source = st.sidebar.radio("Fonte de vídeo", ('Vídeo de exemplo', 'Webcam', 'Outro vídeo'))

    if video_source == 'Vídeo de exemplo':
        video_file = 'videos/exemplo.mp4'
    elif video_source == 'Webcam':
        video_file = 0  # Webcam
    else:
        uploaded_file = st.sidebar.file_uploader("Carregar um vídeo", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            video_file = os.path.join('videos', uploaded_file.name)
            with open(video_file, 'wb') as f:
                f.write(uploaded_file.read())
        else:
            st.warning("Por favor, carregue um arquivo de vídeo.")
            st.stop()

    # Botões de iniciar e parar inferência
    start_inference = st.sidebar.button("Iniciar Inferência", key="start")
    stop_inference = st.sidebar.button("Parar Inferência", key="stop")

    # Espaço para exibir o vídeo
    FRAME_WINDOW = st.empty()

    # Variável para controlar a inferência
    if 'inference_started' not in st.session_state:
        st.session_state['inference_started'] = False

    # Processamento de vídeo
    if start_inference:
        st.session_state['inference_started'] = True
    if stop_inference:
        st.session_state['inference_started'] = False

    if st.session_state['inference_started']:
        cap = cv2.VideoCapture(video_file)
        prev_time = time.time()
        while st.session_state['inference_started']:
            ret, frame = cap.read()
            if not ret:
                st.session_state['inference_started'] = False
                cap.release()
                break

            # Inferência
            results = model.predict(frame, conf=confidence_threshold, verbose=False)

            # Filtrar classes
            detections = results[0]
            if len(detections) > 0 and selected_classes:
                classes = detections.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[c] for c in classes]
                mask = np.isin(class_names, selected_classes)
                detections.boxes = detections.boxes[mask]

            # Anotar o frame
            annotated_frame = detections.plot()

            # Exibir FPS se selecionado
            if display_fps:
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Converter para RGB e exibir
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            # Permitir que o Streamlit atualize a interface
            # e responda a interações do usuário
            if not st.session_state['inference_started']:
                cap.release()
                break

        cap.release()

    else:
        # Exibir o vídeo inicial
        if video_source in ['Vídeo de exemplo', 'Outro vídeo']:
            if os.path.exists(video_file):
                st.video(video_file, start_time=0, format='video/mp4')
            else:
                st.warning(f"Vídeo não encontrado no caminho: {video_file}")
        elif video_source == 'Webcam':
            st.warning("Clique em 'Iniciar Inferência' para usar a Webcam.")
else:
    st.error(f"Nenhum modelo encontrado na pasta {weights_dir}. Por favor, coloque seus modelos lá.")
