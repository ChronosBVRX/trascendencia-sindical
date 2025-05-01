import os
if not os.path.exists("vectorstore/index.faiss"):
    from generar_vectorstore import generar_y_guardar_vectorstore
    generar_y_guardar_vectorstore()
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embedding_service import load_embeddings, buscar_respuesta
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# CORS para que funcione desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o especifica ["https://tuweb.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los embeddings una sola vez al iniciar
print("📦 Cargando base de datos vectorial...")
vectorstore = load_embeddings("vectorstore")
print(f"✅ {len(vectorstore)} fragmentos cargados.")

class Solicitud(BaseModel):
    nombre: str
    cargo: str
    destinatario: str
    problema: str
    peticion: str

@app.post("/generar-escrito")
async def generar_escrito(solicitud: Solicitud):
    try:
        pregunta = f"{solicitud.problema}\n\n{solicitud.peticion}"
        respuesta = buscar_respuesta(pregunta, vectorstore)
        return {"texto": respuesta}
    except Exception as e:
        print(f"💥 Error en /generar-escrito: {e}")
        return {"texto": "Ocurrió un error al generar tu respuesta. Intenta más tarde o revisa tu conexión."}
