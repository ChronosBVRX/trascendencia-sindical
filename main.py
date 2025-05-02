import os
import re
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

load_dotenv()
app = FastAPI()

BASE = os.getcwd()

# Monta los estáticos
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE, "static")),
    name="static"
)

# Sirve index.html en la raíz
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

# Endpoint de consulta
@app.post("/consulta")
async def endpoint_consulta(payload: dict):
    pregunta = (payload.get("texto") or "").strip()
    if not pregunta:
        return {"respuesta": "❗ Oops, parece que no escribiste nada. ¿En qué puedo ayudarte?"}

    # 1) Detección de saludos genéricos
    saludo_pattern = r'^(hola|buenos días|buenas tardes|buenas noches|qué tal|hey)\b'
    if re.match(saludo_pattern, pregunta, re.I):
        return {"respuesta": "¡Hola! 😊 ¿Cómo estás? ¿En qué te puedo apoyar hoy con tu Contrato Colectivo del IMSS?"}

    # 2) Pasar a consulta de cláusulas
    try:
        respuesta = consulta_contrato(pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"¡Uy! Tuve un error interno: {e}"}

# Startup: genera vectorstore si falta
VECTORSTORE_DIR  = os.path.join(BASE, "vectorstore")
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
PICKLE_PATH      = os.path.join(VECTORSTORE_DIR, "index.pkl")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()
