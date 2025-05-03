import os
import fitz
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
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
    # 1) Carga FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 2) Recupera top-5 fragmentos
    top_docs = db.similarity_search(question, k=5)
    context = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # 3) Arma lista de mensajes conservando el hilo
    messages = [SystemMessage(content="""
Eres un asistente experto en el Contrato Colectivo de Trabajo del IMSS.
Habla de forma creativa y natural, como si conversases con un colega.
Cuando cites:
- Menciona la sección (ej. \"Reglamento Interior de Trabajo\").
- Indica el número exacto de cláusula/artículo.
- Extrae el texto **literalmente**.
- No inventes referencias; si no existe, di «No se encontró referencia exacta en el contrato.»
""".strip())]

    # Reconstruye el diálogo previo
    for msg in history:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))

    # Añade el contexto y la pregunta actual
    messages.append(HumanMessage(content=f"Contexto:\n{context}\n\nPregunta: {question}"))

    # 4) Llamada al modelo
    chat = ChatOpenAI(temperature=0.4)
    respuesta = chat(messages).content
    return respuesta
