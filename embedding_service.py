import os
import fitz
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# …tus imports…
HERE = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER        = os.path.join(HERE, "pdfs")        # si pdfs/ está junto a embedding_service.py
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")
PICKLE_PATH        = os.path.join(VECTORSTORE_FOLDER, "index.pkl")

# Carga clave de API
load_dotenv()

# Rutas relativas
ROOT = os.path.dirname(__file__)
PDF_FOLDER = os.path.abspath(os.path.join(ROOT, "..", "pdfs"))
VECTORSTORE_FOLDER = os.path.abspath(os.path.join(ROOT, "..", "vectorstore"))
PICKLE_PATH = os.path.join(VECTORSTORE_FOLDER, "index.pkl")


def cargar_pdfs():
    textos = []
    for f in os.listdir(PDF_FOLDER):
        if f.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(PDF_FOLDER, f))
            textos.append("".join([p.get_text() for p in doc]))
    return textos


def generar_y_guardar_vectorstore():
    textos = cargar_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for t in textos:
        docs.extend(splitter.create_documents([t]))

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db.save_local(VECTORSTORE_FOLDER)

    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(db, f)


def consulta_contrato(pregunta: str) -> str:
    db = FAISS.load_local(VECTORSTORE_FOLDER, OpenAIEmbeddings())
    resultados = db.similarity_search(pregunta, k=5)
    context = "\n".join([doc.page_content for doc in resultados])

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    chat = ChatOpenAI(temperature=0)
    msgs = [
        SystemMessage(content="Asistente para contrato colectivo IMSS."),
        HumanMessage(content=f"Contexto:\n{context}\nPregunta: {pregunta}")
    ]
    return chat(msgs).content
