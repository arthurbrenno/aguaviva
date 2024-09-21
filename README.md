### **Executando o Aplicativo com Docker**

Execute o seguinte comando para iniciar o aplicativo:

```bash
docker-compose up
```


#### **12.4. Acessando a Interface Web**

Abra o navegador e acesse `http://localhost:5000`. Você deverá ver a interface web com o vídeo em tempo real exibindo as detecções de lixo aquático.








### **Testando Sem Docker**

Se preferir testar o aplicativo localmente sem Docker, siga estes passos:

#### **Configurando o Ambiente Virtual (Opcional)**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

#### **Instalando as Dependências**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Executando o Aplicativo**

```bash
python app.py
```

#### **Acessando a Interface Web**

Abra o navegador e vá para `http://localhost:5000`.
