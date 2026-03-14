import os
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Rutas absolutas para evitar problemas al ejecutar desde distintos directorios
HERE = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(HERE, "pdfs")
VECTORSTORE_FOLDER = os.path.join(HERE, "vectorstore")

def cargar_pdfs() -> List[str]:
    """Lee todos los PDFs de la carpeta /pdfs y devuelve una lista con todo su texto."""
    textos = []
    if not os.path.exists(PDF_FOLDER):
        print(f"Advertencia: La carpeta {PDF_FOLDER} no existe. Creándola...")
        os.makedirs(PDF_FOLDER, exist_ok=True)
        return textos

    for fname in os.listdir(PDF_FOLDER):
        if fname.lower().endswith(".pdf"):
            ruta_pdf = os.path.join(PDF_FOLDER, fname)
            try:
                doc = fitz.open(ruta_pdf)
                contenido = "".join(page.get_text() for page in doc)
                textos.append(contenido)
            except Exception as e:
                print(f"Error al leer {fname}: {e}")
    return textos

def generar_y_guardar_vectorstore() -> None:
    """Carga los PDFs, divide el texto en fragmentos, genera embeddings y guarda la BD FAISS."""
    print("Iniciando el procesamiento de PDFs...")
    textos = cargar_pdfs()
    
    if not textos:
        print("No se encontraron PDFs o están vacíos. No se generará el vectorstore.")
        return

    # Dividimos el texto en fragmentos manejables con un poco de superposición para no cortar ideas
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for t in textos:
        docs.extend(splitter.create_documents([t]))

    print(f"Generando embeddings para {len(docs)} fragmentos de texto...")
    embeddings = OpenAIEmbeddings()
    
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_FOLDER)
    print("Vectorstore generado y guardado exitosamente en:", VECTORSTORE_FOLDER)

def consulta_contrato(question: str, history: List[dict]) -> str:
    """Ejecuta una Cadena RAG estricta para garantizar que el bot no invente respuestas."""
    
    # --- 1) CARGAR LA BASE DE DATOS Y CONFIGURAR EL RECUPERADOR ---
    embeddings = OpenAIEmbeddings()
    try:
        db = FAISS.load_local(
            VECTORSTORE_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        return "⚠️ Error: No se encontró la base de datos de documentos. Por favor, reinicia la aplicación para generarla."
        
    # Extraemos 6 fragmentos relevantes para tener suficiente contexto
    retriever = db.as_retriever(search_kwargs={"k": 6})
    
    # Modelo principal configurado con temperatura 0.0 para anular la "creatividad"
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

    # --- 2) PREPARAR EL HISTORIAL DE CHAT ---
    chat_history = []
    for msg in history:
        # Asegurarnos de no procesar la última pregunta como parte del historial previo
        if msg.get("content") == question:
            continue
            
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            chat_history.append(AIMessage(content=msg.get("content", "")))

    # --- 3) CADENA PARA REFORMULAR LA PREGUNTA (CON HISTORIAL) ---
    contextualize_q_system_prompt = (
        "Dado el historial de chat y la última pregunta del usuario, "
        "que podría hacer referencia a contexto previo, formula una pregunta independiente "
        "que se entienda por sí sola sin el historial. NO respondas a la pregunta, "
        "solo reformúlala o devuélvela tal cual si no necesita cambios."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- 4) PROMPT DEL SISTEMA ULTRA-ESTRICTO ---
    qa_system_prompt = """Eres un asesor legal laboral experto en el Contrato Colectivo de Trabajo (CCT) del IMSS y sus reglamentos.
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS E INQUEBRANTABLES:
1. CERO INVENTOS: Tu respuesta debe basarse ÚNICA Y EXCLUSIVAMENTE en el contexto recuperado que se te proporciona más abajo. No uses tu conocimiento general de internet ni asumas nada.
2. MANEJO DE VACÍOS: Si el contexto proporcionado no contiene la respuesta exacta a la pregunta, TIENES PROHIBIDO inventar, deducir o suponer. Debes responder textualmente: «No encontré la referencia exacta para esta consulta en los documentos del CCT.»
3. CITAS PRECISAS: Siempre que fundamentes tu respuesta, cita la fuente indicando la cláusula, artículo o sección que aparece en el contexto (ej. "De acuerdo con la Cláusula X...").
4. FORMATO Y TONO: Responde de forma concisa. Usa un máximo de tres viñetas o ideas breves en un lenguaje accesible. Mantén un tono profesional, empático e institucional.

CONTEXTO RECUPERADO DE LOS DOCUMENTOS:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- 5) CREAR Y EJECUTAR LA CADENA RAG COMPLETA ---
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        resultado = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return resultado["answer"]
    except Exception as e:
        return f"Lo siento, hubo un problema al consultar los documentos: {str(e)}"
