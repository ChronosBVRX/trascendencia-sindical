import os
import fitz
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Importaciones actualizadas: La forma moderna con LangGraph
from langchain_core.tools import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent

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
    """Corrige ortografía, configura herramientas y ejecuta el Agente LangGraph."""
    
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
    question_corr = ortho_model.invoke(ortho_msgs).content.strip()

    # --- 2) CONFIGURACIÓN DE HERRAMIENTAS ---
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    herramienta_cct = create_retriever_tool(
        retriever,
        "buscar_contrato_colectivo",
        "Usa ESTA herramienta SIEMPRE en primer lugar para buscar sobre derechos, obligaciones, cláusulas o reglamentos del IMSS."
    )

    herramienta_internet = DuckDuckGoSearchResults(
        name="buscar_leyes_externas",
        description="Usa esta herramienta ÚNICAMENTE si el usuario pregunta específicamente por leyes externas como la Ley Federal del Trabajo, Ley General de Salud, o la Constitución Mexicana."
    )

    tools = [herramienta_cct, herramienta_internet]

    # --- 3) PROMPT DEL SISTEMA ---
    system_message = """
Eres un asesor legal laboral experto en el Contrato Colectivo de Trabajo (CCT) del IMSS y sus reglamentos.
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS DE RESPUESTA:
1. PRIORIDAD CCT: Si la pregunta es sobre el IMSS, usa SIEMPRE la herramienta 'buscar_contrato_colectivo'.
2. LEYES EXTERNAS: Si el usuario menciona explícitamente leyes externas (LFT, Ley General de Salud), usa 'buscar_leyes_externas'.
3. CERO INVENTOS: Responde ÚNICAMENTE utilizando la información obtenida de las herramientas. Si no encuentras la respuesta, di: «No se encontró la referencia exacta para esta consulta específica.»
4. CITAS PRECISAS: Siempre que fundamentes tu respuesta, cita la fuente exacta (Ej. «Cláusula X del CCT» o «Artículo X de la LFT»).
5. FORMATO: Responde de forma concisa. Usa un máximo de tres viñetas o ideas breves en un lenguaje accesible.
6. TONO: Mantén un tono profesional, institucional pero empático y cercano.
"""

    # --- 4) PREPARAR EL HISTORIAL PARA LANGGRAPH ---
    chat_history = []
    # Pasamos todo el historial excepto la última pregunta sin corregir
    for msg in history[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
            
    # Añadimos la pregunta ya corregida al final del historial
    chat_history.append(HumanMessage(content=question_corr))

    # --- 5) EJECUCIÓN DEL AGENTE CON LANGGRAPH ---
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini") 
    
    # LangGraph es la forma moderna y recomendada de crear agentes
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

    try:
        # LangGraph requiere un diccionario con la clave "messages"
        resultado = agent_executor.invoke({"messages": chat_history})
        # El resultado devuelve toda la lista de mensajes, tomamos el último (la respuesta final de la IA)
        return resultado["messages"][-1].content
    except Exception as e:
        return f"Lo siento, hubo un problema al consultar los documentos: {str(e)}"
