```python
# main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Carga variables de entorno
load_dotenv()

# Importa funciones de embedding_service
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

app = FastAPI()

# Rutas de los archivos de índice
VECTORSTORE_PATH = "../vectorstore/index.faiss"
PICKLE_PATH = "../vectorstore/index.pkl"

@app.on_event("startup")
def startup_event():
    # Si no existen, crea el índice desde PDFs
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()

@app.post("/consulta")
def endpoint_consulta(payload: dict):
    pregunta = payload.get("texto", "")
    respuesta = consulta_contrato(pregunta)
    return {"respuesta": respuesta}
```
