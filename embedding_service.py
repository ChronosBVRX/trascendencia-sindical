# embedding_service.py

import os
import fitz       # PyMuPDF para leer PDFs
import pickle     # Para serializar el objeto FAISS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carga variables de entorno, especialmente OPENAI_API_KEY
load_dotenv()

# Rutas: asumimos que pdfs/ y vectorstore/ están junto a este archivo
HERE                = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER          = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER  = os.path.join(HERE, "vectorstore")
PICKLE_PATH         = os.path.join(VECTORSTORE_FOLDER, "index.pkl")


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
    4) Crea y guarda el índice FAISS en disk
    5) Serializa opcionalmente con pickle
    """
    # 1) Cargar textos
    textos = cargar_pdfs()

    # 2) Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for texto in textos:
        docs.extend(splitter.create_documents([texto]))

    # 3) Embeddings
    embeddings = OpenAIEmbeddings()

    # 4) Generar FAISS
    db = FAISS.from_documents(docs, embeddings)

    # Asegurar carpeta
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

    # Guardar FAISS en disk
    db.save_local(VECTORSTORE_FOLDER)

    # 5) Serializar con pickle (por si lo necesitas)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(db, f)


def consulta_contrato(pregunta: str) -> str:
    """
    1) Carga el índice FAISS
    2) Busca los 5 documentos más similares
    3) Concatena su texto como contexto
    4) Genera una respuesta con ChatOpenAI usando ese contexto
    """
    # Cargar FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(VECTORSTORE_FOLDER, embeddings)

    # Similarity search
    top_docs = db.similarity_search(pregunta, k=5)
    context = "\n".join(doc.page_content for doc in top_docs)

    # Generar respuesta
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="Eres un asistente que ayuda a trabajadores del IMSS a consultar su contrato colectivo."),
        HumanMessage(content=f"Contexto:\n{context}\nPregunta: {pregunta}")
    ]
    respuesta = chat(messages).content
    return respuesta
