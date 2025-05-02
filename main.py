from fastapi import FastAPI
from dotenv import load_dotenv
import os
import fitz
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # fija el import

load_dotenv()

# Aseguramos usar la carpeta donde arranca uvicorn (pwd)
BASE = os.getcwd()

PDF_FOLDER = os.path.join(BASE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(BASE, "vectorstore")
PICKLE_PATH = os.path.join(VECTORSTORE_FOLDER, "index.pkl")

# Carga variables de entorno desde .env
load_dotenv()

# Importa las funciones del servicio de embeddings
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

app = FastAPI()

# Rutas de índice
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index.faiss")
PICKLE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index.pkl")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()

@app.post("/consulta")
def endpoint_consulta(payload: dict):
    pregunta = payload.get("texto", "")
    respuesta = consulta_contrato(pregunta)
    return {"respuesta": respuesta}
