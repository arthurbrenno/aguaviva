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
