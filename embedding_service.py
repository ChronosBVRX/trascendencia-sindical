import os
import fitz
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

HERE               = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER         = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")


def cargar_pdfs() -> List[str]:
    textos = []
    for f in os.listdir(PDF_FOLDER):
        if f.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(PDF_FOLDER, f))
            textos.append("".join(page.get_text() for page in doc))
    return textos


def generar_y_guardar_vectorstore() -> None:
    textos = cargar_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for t in textos:
        docs.extend(splitter.create_documents([t]))
    embeddings = OpenAIEmbeddings()
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_FOLDER)


def consulta_contrato(question: str, history: List[dict]) -> str:
    # Carga FAISS con deserialización permitida
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Top 5 fragmentos
    resultados = db.similarity_search(question, k=5)
    contexto = "\n".join(f"— Fragmento:\n{d.page_content}" for d in resultados)

    # Prompt reforzado
    system_prompt = """
Eres un asistente experto en el Contrato Colectivo de Trabajo del IMSS.
Habla de forma creativa y natural, como si conversases con un colega.
Al responder:
1) Menciona la sección (ej. "Reglamento Interior de Trabajo").
2) Indica el número exacto de cláusula o artículo.
3) Extrae el texto literalmente tal como aparece.
4) Si no localizas la referencia exacta, di «No se encontró referencia exacta en el contrato.»
""".strip()

    # Reconstruye el diálogo previo
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Añade contexto y pregunta actual
    messages.append(HumanMessage(content=f"Contexto:\n{contexto}\n\nPregunta: {question}"))

    chat = ChatOpenAI(temperature=0.4)
    respuesta = chat(messages).content
    return respuesta
