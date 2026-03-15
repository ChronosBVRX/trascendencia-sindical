import os
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- IMPORTACIONES MODERNAS (Adiós a langchain.chains) ---
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent

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
    """Ejecuta una consulta estricta con LangGraph para garantizar que el bot no invente respuestas."""
    
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
    
    # --- 2) CONVERTIR EL RECUPERADOR EN UNA HERRAMIENTA ---
    herramienta_cct = create_retriever_tool(
        retriever,
        "buscar_contrato_colectivo",
        "Usa ESTA herramienta SIEMPRE para buscar sobre derechos, obligaciones o cláusulas del IMSS."
    )
    tools = [herramienta_cct]

    # --- 3) PREPARAR EL MODELO Y EL AGENTE ---
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    agent_executor = create_react_agent(llm, tools)

    # --- 4) PROMPT DEL SISTEMA ULTRA-ESTRICTO ---
    system_message = """Eres un asesor legal laboral experto en el Contrato Colectivo de Trabajo (CCT) del IMSS y sus reglamentos.
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS E INQUEBRANTABLES:
1. CERO INVENTOS: Tu respuesta debe basarse ÚNICA Y EXCLUSIVAMENTE en la información obtenida al usar la herramienta 'buscar_contrato_colectivo'. No uses tu conocimiento general ni asumas nada.
2. MANEJO DE VACÍOS: Si la herramienta no devuelve información útil para la pregunta, TIENES PROHIBIDO inventar o deducir. Responde textualmente: «No encontré la referencia exacta para esta consulta en los documentos del CCT.»
3. CITAS PRECISAS: Siempre que fundamentes tu respuesta, cita la fuente indicando la cláusula, artículo o sección (ej. "De acuerdo con la Cláusula X...").
4. FORMATO Y TONO: Responde de forma concisa. Usa un máximo de tres viñetas o ideas breves en un lenguaje accesible. Mantén un tono profesional, empático e institucional."""

    # --- 5) CONSTRUIR EL HISTORIAL PARA EL AGENTE ---
    mensajes_finales = [SystemMessage(content=system_message)]
    
    for msg in history:
        if msg.get("content") == question:
            continue
        if msg.get("role") == "user":
            mensajes_finales.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            mensajes_finales.append(AIMessage(content=msg.get("content", "")))

    # Añadir la pregunta actual al final
    mensajes_finales.append(HumanMessage(content=question))

    # --- 6) EJECUCIÓN DEL AGENTE ---
    try:
        resultado = agent_executor.invoke({"messages": mensajes_finales})
        return resultado["messages"][-1].content
    except Exception as e:
        return f"Lo siento, hubo un problema al consultar los documentos: {str(e)}"
