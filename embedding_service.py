import os
import fitz
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

HERE               = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER         = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")


def cargar_pdfs() -> list[str]:
    textos = []
    for nombre in os.listdir(PDF_FOLDER):
        if nombre.lower().endswith(".pdf"):
            ruta = os.path.join(PDF_FOLDER, nombre)
            doc = fitz.open(ruta)
            contenido = "".join(page.get_text() for page in doc)
            textos.append(contenido)
    return textos


def generar_y_guardar_vectorstore() -> None:
    textos = cargar_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for texto in textos:
        docs.extend(splitter.create_documents([texto]))

    embeddings = OpenAIEmbeddings()
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_FOLDER)


def consulta_contrato(pregunta: str) -> str:
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Recupera los 5 fragmentos más relevantes
    top_docs = db.similarity_search(pregunta, k=5)
    context = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # Prompt reforzado: NO saludar de nuevo, ir directo a sección/artículo y texto literal
    SYSTEM_PROMPT = """
Eres un asistente especializado en el Contrato Colectivo de Trabajo del IMSS.
Habla de forma natural pero **sin** volver a saludar al usuario ni introducir “Hola” de nuevo.
Cuando respondas:
1) Indica la **sección** (por ejemplo “Reglamento Interior de Trabajo”).
2) Indica el **número exacto** de la cláusula o artículo.
3) Extrae el texto **literalmente** tal como aparece en el contexto.
4) Si no encuentras la referencia exacta, responde:
   «No se encontró referencia exacta en el contrato.»
"""

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto:\n{context}\n---\nPregunta: {pregunta}")
    ]

    return chat(messages).content
