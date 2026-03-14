import os
import fitz
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

# Importaciones modernas para Cadenas RAG estrictas
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

HERE               = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER         = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")

def cargar_pdfs() -> List[str]:
    """Lee todos los PDFs de /pdfs y devuelve una lista con todo su texto."""
    textos = []
    for fname in os.listdir(PDF_FOLDER):
        if fname.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(PDF_FOLDER, fname))
            contenido = "".join(page.get_text() for page in doc)
            textos.append(contenido)
    return textos

def generar_y_guardar_vectorstore() -> None:
    """Carga PDFs, divide texto, genera embeddings y guarda FAISS."""
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
    """Ejecuta una Cadena RAG estricta para garantizar que no haya alucinaciones."""
    
    # --- 1) CONFIGURACIÓN DEL RETRIEVER ---
    embeddings = OpenAIEmbeddings()
    try:
        db = FAISS.load_local(
            VECTORSTORE_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        return "Error: No se encontró la base de datos de documentos. Por favor, generala primero."
        
    # k=6 asegura que traiga suficiente contexto de los PDFs
    retriever = db.as_retriever(search_kwargs={"k": 6})
    
    # Usamos temperatura 0.0 para anular la creatividad
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

    # --- 2) PREPARAR EL HISTORIAL DE CHAT ---
    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # --- 3) CADENA PARA REFORMULAR LA PREGUNTA (CON HISTORIAL) ---
    # Esto asegura que si el usuario dice "y qué pasa con eso?", el bot entienda el contexto
    contextualize_q_system_prompt = (
        "Dado el historial de chat y la última pregunta del usuario, "
        "que podría hacer referencia a contexto previo, formula una pregunta independiente "
        "que se entienda por sí sola. NO respondas a la pregunta, solo reformúlala o "
        "devuélvela tal cual si no necesita cambios."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- 4) PROMPT DEL SISTEMA ULTRA-ESTRICTO (QA) ---
    qa_system_prompt = """Eres un asesor legal laboral experto en el Contrato Colectivo de Trabajo (CCT) del IMSS y sus reglamentos.
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS E INQUEBRANTABLES:
1. CERO INVENTOS: Tu respuesta debe basarse ÚNICA Y EXCLUSIVAMENTE en el contexto recuperado que se te proporciona más abajo. No uses tu conocimiento general.
2. MANEJO DE VACÍOS: Si el contexto proporcionado no contiene la respuesta a la pregunta, TIENES PROHIBIDO inventar, deducir o suponer. Debes responder textualmente: «No encontré la referencia exacta para esta consulta en los documentos del CCT.»
3. CITAS PRECISAS: Siempre que fundamentes tu respuesta, cita la fuente indicando la cláusula, artículo o sección que aparece en el contexto.
4. FORMATO Y TONO: Responde de forma concisa. Usa un máximo de tres viñetas o ideas breves. Mantén un tono profesional, empático e institucional.

CONTEXTO RECUPERADO DE LOS DOCUMENTOS:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- 5) CREAR Y EJECUTAR LA CADENA RAG ---
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        # Ejecutamos la cadena completa
        resultado = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return resultado["answer"]
    except Exception as e:
        return f"Lo siento, hubo un problema al consultar los documentos: {str(e)}"
