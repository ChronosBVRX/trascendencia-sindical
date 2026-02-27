import os
import fitz
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Nuevas importaciones para el Agente y las Herramientas
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    4) Guarda el índice en disco
    """
    textos = cargar_pdfs()
    # Se mantiene el tamaño de chunk que tenías, es adecuado para textos legales
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
    2) Configura herramientas (FAISS y Buscador web)
    3) Ejecuta el Agente para responder con precisión
    """
    # --- 1) CORRECCIÓN ORTOGRÁFICA SILENCIOSA ---
    # Usamos invoke() que es el método actualizado en LangChain
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
    # Convertimos FAISS en un recuperador (retriever)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Herramienta 1: Base de datos interna (CCT)
    herramienta_cct = create_retriever_tool(
        retriever,
        "buscar_contrato_colectivo",
        "Usa ESTA herramienta SIEMPRE en primer lugar para buscar sobre derechos, obligaciones, cláusulas o reglamentos del IMSS."
    )

    # Herramienta 2: Buscador en Internet (Leyes externas)
    herramienta_internet = DuckDuckGoSearchResults(
        name="buscar_leyes_externas",
        description="Usa esta herramienta ÚNICAMENTE si el usuario pregunta específicamente por leyes externas como la Ley Federal del Trabajo, Ley General de Salud, o la Constitución Mexicana."
    )

    tools = [herramienta_cct, herramienta_internet]

    # --- 3) PROMPT DEL AGENTE ---
    prompt_agente = ChatPromptTemplate.from_messages([
        ("system", """
Eres un asesor legal laboral experto en el Contrato Colectivo de Trabajo (CCT) del IMSS y sus reglamentos.
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS DE RESPUESTA:
1. PRIORIDAD CCT: Si la pregunta es sobre el IMSS, usa SIEMPRE la herramienta 'buscar_contrato_colectivo'.
2. LEYES EXTERNAS: Si el usuario menciona explícitamente leyes externas (LFT, Ley General de Salud), usa 'buscar_leyes_externas'.
3. CERO INVENTOS: Responde ÚNICAMENTE utilizando la información obtenida de las herramientas. Si no encuentras la respuesta, di: «No se encontró la referencia exacta para esta consulta específica.»
4. CITAS PRECISAS: Siempre que fundamentes tu respuesta, cita la fuente exacta (Ej. «Cláusula X del CCT» o «Artículo X de la LFT»).
5. FORMATO: Responde de forma concisa. Usa un máximo de tres viñetas o ideas breves en un lenguaje accesible.
6. TONO: Mantén un tono profesional, institucional pero empático y cercano.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- 4) PREPARAR EL HISTORIAL ---
    # Convertimos los diccionarios que vienen de main.py a objetos Message de LangChain
    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # --- 5) EJECUCIÓN DEL AGENTE ---
    # gpt-4o-mini es excelente y muy económico para seguir instrucciones de herramientas
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini") 
    agent = create_openai_tools_agent(llm, tools, prompt_agente)
    
    # verbose=False para no saturar los logs de Render, puedes cambiar a True si necesitas depurar
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    try:
        resultado = agent_executor.invoke({
            "input": question_corr,
            "chat_history": chat_history
        })
        return resultado["output"]
    except Exception as e:
        return f"Lo siento, hubo un problema al consultar los documentos: {str(e)}"
