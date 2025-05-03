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
    """
    Lee todos los PDFs de /pdfs y devuelve una lista con todo su texto.
    """
    textos = []
    for fname in os.listdir(PDF_FOLDER):
        if fname.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(PDF_FOLDER, fname))
            contenido = "".join(page.get_text() for page in doc)
            textos.append(contenido)
    return textos


def generar_y_guardar_vectorstore() -> None:
    """
    1) Carga los PDFs
    2) Divide el texto en trozos
    3) Genera embeddings y construye FAISS
    4) Guarda el índice en disk
    """
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
    """
    1) Corrige ortografía de la pregunta
    2) Busca en FAISS usando la versión corregida
    3) Genera respuesta creativa, citando sección y artículo/cláusula
    """
    # --- 1) CORRECCIÓN ORTOGRÁFICA SILENCIOSA ---
    ortho_model = ChatOpenAI(temperature=0)
    ortho_msgs = [
        SystemMessage(content=(
            "Eres un corrector ortográfico: recibe una pregunta del usuario y devuelve "
            "solo la versión corregida con gramática y ortografía apropiadas, "
            "sin añadir ni quitar contenido."
        )),
        HumanMessage(content=question)
    ]
    question_corr = ortho_model(ortho_msgs).content.strip()

    # --- 2) BÚSQUEDA EN FAISS ---
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    top_docs = db.similarity_search(question_corr, k=5)
    contexto = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # --- 3) RESPUESTA FINAL ---
    system_prompt = """
Eres un asistente experto en el Contrato Colectivo de Trabajo del IMSS.
Habla de forma creativa y natural, como si conversarás con un amigo.
Al responder:
1) Menciona la sección exacta (por ejemplo "Reglamento Interior de Trabajo").
2) Indica el número exacto de cláusula o artículo.
3) Extrae **literalmente** el texto relevante.
4) Si no localizas la referencia exacta, di:
   «No se encontró referencia exacta en el contrato.»
5) Recuerda que la pregunta pudo venir con faltas de ortografía; interpreta usando la versión corregida internamente.
""".strip()

    # Reconstruye el hilo completo
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Añade contexto y pregunta corregida
    messages.append(HumanMessage(content=(
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta (corregida): {question_corr}"
    )))

    chat = ChatOpenAI(temperature=0.4)
    return chat(messages).content
