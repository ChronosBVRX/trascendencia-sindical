import os
import re
from typing import List, Literal
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

load_dotenv()
app = FastAPI()
BASE = os.getcwd()

# Montamos estáticos en /static
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE, "static")),
    name="static"
)

# Sirve la UI
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

# Modelos Pydantic para validar la petición
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ConsultaRequest(BaseModel):
    history: List[Message]

@app.post("/consulta")
async def endpoint_consulta(req: ConsultaRequest):
    history = req.history
    if not history:
        return {"respuesta": "❗ No recibí ninguna pregunta. ¿En qué puedo ayudar?"}

    # Si el usuario solo saludó, devolvemos la presentación experta
    if len(history) == 1 and history[0].role == "user":
        saludo = history[0].content.strip()
        if re.match(r'^(hola|buenos días|buenas tardes|buenas noches|hey|qué tal)\s*$', saludo, re.I):
            return {
                "respuesta": (
                    "¡Hola! 👋 Soy tu asistente experto en temas contractuales y sindicales del IMSS. "
                    "Listo para ayudarte con cualquier consulta de tu Contrato Colectivo. ¿En qué puedo servirte hoy?"
                )
            }

    # El “question” es el último mensaje del usuario
    question = None
    # Encuentra el último mensaje con role="user"
    for msg in reversed(history):
        if msg.role == "user":
            question = msg.content.strip()
            break

    if not question:
        return {"respuesta": "❗ No pude encontrar tu pregunta en el historial."}

    try:
        respuesta = consulta_contrato(question, history)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"¡Uy! Error interno: {e}"}

# Al arrancar, generamos vectorstore si falta
VECTORSTORE_DIR  = os.path.join(BASE, "vectorstore")
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
PICKLE_PATH      = os.path.join(VECTORSTORE_DIR, "index.pkl")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()
