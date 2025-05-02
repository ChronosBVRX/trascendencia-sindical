# embedding_service.py

import os
import fitz                        # PyMuPDF para leer PDFs
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carga tu OPENAI_API_KEY desde .env
load_dotenv()

# Rutas: asumimos que pdfs/ y vectorstore/ están junto a este archivo
HERE               = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER         = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")


def cargar_pdfs() -> list[str]:
    """
    Lee todos los PDFs de la carpeta 'pdfs/' y devuelve una lista con su texto completo.
    """
    textos = []
    for nombre in os.listdir(PDF_FOLDER):
        if nombre.lower().endswith(".pdf"):
            ruta = os.path.join(PDF_FOLDER, nombre)
            doc = fitz.open(ruta)
            contenido = "".join(page.get_text() for page in doc)
            textos.append(contenido)
    return textos


def generar_y_guardar_vectorstore() -> None:
    """
    1) Carga los PDFs
    2) Divide el texto en fragments (chunks)
    3) Genera embeddings con OpenAI
    4) Crea y guarda el índice FAISS en disco
    """
    # 1) Cargar todos los textos
    textos = cargar_pdfs()

    # 2) Chunking: 1000 caracteres con 200 de solapamiento
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for texto in textos:
        docs.extend(splitter.create_documents([texto]))

    # 3) Generar embeddings
    embeddings = OpenAIEmbeddings()

    # 4) Construir FAISS y guardar en disco
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_FOLDER)


def consulta_contrato(pregunta: str) -> str:
    """
    1) Carga el índice FAISS
    2) Busca los 5 documentos más similares
    3) Usa su texto como contexto para ChatOpenAI
    4) Responde SÓLO con información literal o indica que no la encuentra
    """
    # Cargar embeddings e índice FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(VECTORSTORE_FOLDER, embeddings)

    # 1) Búsqueda de similitud
    top_docs = db.similarity_search(pregunta, k=5)
    context = "\n".join(f"— Fragmento:\n{doc.page_content}" for doc in top_docs)

    # 2) Prompt del sistema reforzado
    SYSTEM_PROMPT = """
Eres un asistente legal especializado en el Contrato Colectivo de Trabajo del IMSS.
– RESPONDE SÓLO con datos que aparezcan LITERALMENTE en el contexto proporcionado.
– INDICA el número exacto de la cláusula y EXTRAE el texto tal cual aparece.
– Si NO encuentras la referencia, di «No se encontró referencia exacta en el contrato».
"""

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto:\n{context}\n---\nPregunta: {pregunta}")
    ]

    respuesta = chat(messages).content
    return respuesta
