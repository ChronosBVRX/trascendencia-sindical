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

    # Buscamos los 5 fragmentos más relevantes
    top_docs = db.similarity_search(pregunta, k=5)
    context = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # Prompt reforzado con tono conversacional
    SYSTEM_PROMPT = """
Eres un asistente legal especializado en el Contrato Colectivo de Trabajo del IMSS.
Habla siempre de forma conversacional, cálida y natural, como si platicaras con un colega.
– RESPONDE solo con información que aparezca LITERALMENTE en el contexto.
– INDICA el número exacto de la cláusula y EXTRAÉ el texto tal cual.
– Si NO localizas la referencia exacta, responde «No se encontró referencia exacta en el contrato.»
"""

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto:\n{context}\n---\nPregunta: {pregunta}")
    ]

    return chat(messages).content
