import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_service import load_embeddings, buscar_respuesta
from generar_vectorstore import generar_y_guardar_vectorstore

app = FastAPI()
vectorstore = None

# Modelo para recibir preguntas
class PreguntaInput(BaseModel):
    pregunta: str

@app.on_event("startup")
def cargar_vectorstore():
    global vectorstore
    if not os.path.exists("vectorstore/index.faiss") or not os.path.exists("vectorstore/index.pkl"):
        print("⚠️ No se encontró vectorstore, generando uno nuevo...")
        generar_y_guardar_vectorstore()

    print("📦 Cargando base de datos vectorial...")
    try:
        vectorstore = load_embeddings("vectorstore")
        print("✅ Vectorstore cargado con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar vectorstore: {e}")
        raise e

@app.post("/preguntar")
def responder_pregunta(datos: PreguntaInput):
    global vectorstore
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore no está disponible.")
    
    respuesta = buscar_respuesta(datos.pregunta, vectorstore)
    return {"respuesta": respuesta}
