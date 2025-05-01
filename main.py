from fastapi import FastAPI, Request
from embedding_service import load_embeddings, buscar_respuesta
from pdf_loader import procesar_pdfs
import os

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    if not os.path.exists("vector_data.json"):
        procesar_pdfs("pdfs/")
        load_embeddings()

@app.post("/preguntar")
async def preguntar(req: Request):
    datos = await req.json()
    pregunta = datos.get("pregunta")
    respuesta = buscar_respuesta(pregunta)
    return {"respuesta": respuesta}
