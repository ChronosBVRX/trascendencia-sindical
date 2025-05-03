import os
import re
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
    """
    Lee todos los PDFs de pdfs/ y devuelve una lista con su texto completo.
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
    2) Divide el texto en chunks
    3) Genera embeddings
    4) Guarda el índice FAISS en disco
    """
    textos = cargar_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for texto in textos:
        docs.extend(splitter.create_documents([texto]))

    embeddings = OpenAIEmbeddings()
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_FOLDER)


def _fallback_buscar_vacaciones() -> str:
    """
    Si la consulta trata de 'vacaciones' y FAISS no devuelve nada,
    busca directamente en el texto completo del PDF.
    """
    for texto in cargar_pdfs():
        # Busca 'vacacion' en todo el texto, capturando cláusula/artículo
        m = re.search(
            r'(Cláusula|Artículo)\s*(\d+)[\s\S]{0,200}?vacaciones[^\n]*',
            texto,
            flags=re.IGNORECASE
        )
        if m:
            # Extraemos el fragmento completo de esa cláusula
            fragmento = m.group(0).strip()
            seccion = "Reglamento Interior de Trabajo"  # Ajusta si tu PDF tiene otro nombre de sección
            return (
                f"Sección: {seccion}\n"
                f"{m.group(1)} {m.group(2)}:\n\"{fragmento}\""
            )
    return "No se encontró referencia exacta en el contrato."


def consulta_contrato(pregunta: str) -> str:
    """
    1) Intenta con FAISS
    2) Si la pregunta contiene 'vacaciones' y no hay resultado, usa fallback
    """
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 1) Similarity search
    top_docs = db.similarity_search(pregunta, k=5)
    # Unir contenidos
    context = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # 2) Prompt reforzado
    SYSTEM_PROMPT = """
Eres un asistente especializado en el Contrato Colectivo de Trabajo del IMSS.
— Habla de forma natural y directa, sin volver a saludar.
— Indica la sección, el número exacto de cláusula/artículo y extrae el texto literal.
— Si no localizas referencia exacta, responde «No se encontró referencia exacta en el contrato.»
"""

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto:\n{context}\n---\nPregunta: {pregunta}")
    ]
    respuesta = chat(messages).content

    # 3) Si no se halló nada y la pregunta es sobre vacaciones, fallback
    if re.search(r'vacacion', pregunta, re.IGNORECASE) and \
       "No se encontró referencia exacta" in respuesta:
        return _fallback_buscar_vacaciones()

    return respuesta
