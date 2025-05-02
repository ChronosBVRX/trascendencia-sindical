import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

load_dotenv()
app = FastAPI()

# Monta la carpeta static/ en la raíz
app.mount("/", StaticFiles(directory=os.path.join(os.getcwd(), "static"), html=True), name="static")

# Resto de tu configuración…
BASE = os.getcwd()
VECTORSTORE_PATH = os.path.join(BASE, "vectorstore", "index.faiss")
PICKLE_PATH      = os.path.join(BASE, "vectorstore", "index.pkl")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()

@app.post("/consulta")
def endpoint_consulta(payload: dict):
    pregunta = payload.get("texto", "").strip()
    if not pregunta:
        return {"respuesta": "❗ Envía tu pregunta en el campo 'texto'."}
    try:
        return {"respuesta": consulta_contrato(pregunta)}
    except Exception as e:
        return {"error": f"Error interno: {e}"}
