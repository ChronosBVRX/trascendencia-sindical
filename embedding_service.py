import os
import fitz                        # PyMuPDF para leer PDFs
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carga tu OPENAI_API_KEY desde .env
load_dotenv()

# Carpeta donde están tus PDFs (junto a este archivo)
HERE       = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(HERE, "pdfs")

# Variable global para el vectorstore en memoria
_VECTORSTORE_DB = None

def cargar_pdfs() -> list[str]:
    """
    Lee todos los PDFs de pdfs/ y devuelve una lista de textos.
    """
    textos = []
    for nombre in os.listdir(PDF_FOLDER):
        if nombre.lower().endswith(".pdf"):
            ruta = os.path.join(PDF_FOLDER, nombre)
            doc = fitz.open(ruta)
            contenido = "".join(page.get_text() for page in doc)
            textos.append(contenido)
    return textos

def build_vectorstore():
    """
    Crea el FAISS vectorstore en memoria a partir de los PDFs.
    """
    global _VECTORSTORE_DB

    textos = cargar_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for texto in textos:
        docs.extend(splitter.create_documents([texto]))

    embeddings = OpenAIEmbeddings()
    _VECTORSTORE_DB = FAISS.from_documents(docs, embeddings)

def generar_y_guardar_vectorstore() -> None:
    """
    Inicializa el vectorstore en memoria.
    (Ya no guarda nada en disco.)
    """
    build_vectorstore()

def consulta_contrato(pregunta: str) -> str:
    """
    Busca en el vectorstore en memoria y genera la respuesta.
    """
    global _VECTORSTORE_DB
    if _VECTORSTORE_DB is None:
        build_vectorstore()

    db = _VECTORSTORE_DB
    top_docs = db.similarity_search(pregunta, k=5)
    context = "\n".join(doc.page_content for doc in top_docs)

    # Genera la respuesta con ChatOpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="Eres un asistente que ayuda a trabajadores del IMSS a consultar su contrato colectivo de trabajo."),
        HumanMessage(content=f"Contexto:\n{context}\nPregunta: {pregunta}")
    ]
    return chat(messages).content
