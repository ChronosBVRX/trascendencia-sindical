1. **Ubicación**: `project/main.py`
2. **Contenido completo**:
```python
from fastapi import FastAPI
import os

# Importa las funciones que creamos en embedding_service.py
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

# Crea la aplicación FastAPI
app = FastAPI()

# Rutas donde se guardan los archivos generados
VECTORSTORE_PATH = "vectorstore/index.faiss"
PICKLE_PATH = "vectorstore/index.pkl"

@app.on_event("startup")
def startup_event():
    # Al inicio, verifica si existen los archivos de índice
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        # Si faltan, crea el vectorstore desde los PDFs
        generar_y_guardar_vectorstore()

@app.post("/consulta")
def consultar_contrato(payload: dict):
    # Espera un JSON con la clave "texto"
    pregunta = payload.get("texto", "")
    # Llama a la función que busca y responde
    respuesta = consulta_contrato(pregunta)
    return {"respuesta": respuesta}
