import os
import re
from typing import List, Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Importamos la lógica de IA desde tu servicio de embeddings
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

load_dotenv()
BASE = os.getcwd()
VECTORSTORE_DIR  = os.path.join(BASE, "vectorstore")
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
PICKLE_PATH      = os.path.join(VECTORSTORE_DIR, "index.pkl")

# --- MANEJO MODERNO DEL INICIO DE LA APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta al arrancar el servidor
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        print("Vectorstore no encontrado. Generando uno nuevo a partir de los PDFs...")
        generar_y_guardar_vectorstore()
    yield
    # Aquí iría el código de apagado si en el futuro necesitas limpiar recursos

app = FastAPI(lifespan=lifespan)

# Monta estáticos en /static
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE, "static")),
    name="static"
)

# GET / → index.html
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

# HEAD / → también sirve index.html para no 405
@app.head("/")
async def head_index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

# Esquema de mensajes para el historial
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ConsultaRequest(BaseModel):
    history: List[Message]

# POST /consulta
@app.post("/consulta")
async def endpoint_consulta(req: ConsultaRequest):
    history = req.history
    if not history:
        return {"respuesta": "❗ No recibí ninguna pregunta. ¿En qué puedo ayudar?"}

    # Si solo saludan, presentación experta
    if len(history) == 1 and history[0].role == "user":
        saludo = history[0].content.strip()
        if re.match(r'^(hola|buenos días|buenas tardes|buenas noches|hey|qué tal)\s*$', saludo, re.I):
            return {
                "respuesta": (
                    "¡Hola! 👋 Soy tu asistente experto en temas contractuales y sindicales del IMSS. "
                    "Listo para ayudarte con cualquier consulta de tu Contrato Colectivo. ¿En qué puedo servirte hoy?"
                )
            }

    # Extrae la última pregunta del usuario
    question = None
    for msg in reversed(history):
        if msg.role == "user":
            question = msg.content.strip()
            break

    if not question:
        return {"respuesta": "❗ No pude encontrar tu pregunta en el historial."}

    try:
        # Pasa el historial convertido a diccionarios usando el método moderno de Pydantic
        historial_dicts = [h.model_dump() if hasattr(h, 'model_dump') else h.dict() for h in history]
        
        # Llama a la cadena RAG estricta
        respuesta = consulta_contrato(question, historial_dicts)
        return {"respuesta": respuesta}
        
    except Exception as e:
        return {"error": f"¡Uy! Ocurrió un error interno: {str(e)}"}
