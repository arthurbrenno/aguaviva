# Use uma imagem base do Ubuntu
FROM ubuntu:20.04

# Defina variáveis de ambiente para evitar prompts interativos durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Atualize os pacotes e instale dependências do sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    curl \
    supervisor \
    xfce4 \
    xfce4-goodies \
    tigervnc-standalone-server \
    tigervnc-common \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Crie um usuário não root para executar a aplicação
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Mude para o usuário criado
USER docker
WORKDIR /home/docker

# Instale as dependências Python
COPY --chown=docker:docker requirements.txt /home/docker/
RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie o código da aplicação para o contêiner
COPY --chown=docker:docker . /home/docker/

# Configure o Supervisor para gerenciar o VNC e a aplicação
RUN mkdir -p /home/docker/.vnc && \
    echo "docker" | vncpasswd -f > /home/docker/.vnc/passwd && \
    chmod 600 /home/docker/.vnc/passwd

# Crie o arquivo de configuração do Supervisor
COPY --chown=docker:docker supervisord.conf /home/docker/supervisord.conf

# Exponha a porta do VNC
EXPOSE 5901

# Defina o comando de inicialização
CMD ["/usr/bin/supervisord", "-c", "/home/docker/supervisord.conf"]
