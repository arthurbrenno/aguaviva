# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Instale as dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie os arquivos de requisitos
COPY requirements.txt .

# Instale as dependências Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante dos arquivos do aplicativo
COPY . .

# Exponha a porta que o Flask usará
EXPOSE 5000

# Defina a variável de ambiente para o Flask
ENV FLASK_APP=app.py

# Comando para executar o aplicativo
CMD ["python", "app.py"]
