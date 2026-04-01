import os
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- IMPORTACIONES MODERNAS ---
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

    # AUMENTADO: Un poco más de tamaño y overlap para no cortar cláusulas largas a la mitad
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
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
        
    # MEJORA: Uso de MMR para traer fragmentos diversos (no solo los más similares entre sí)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    
    # --- 2) CONVERTIR EL RECUPERADOR EN UNA HERRAMIENTA ---
    # MEJORA: Instrucciones más claras para forzar al LLM a usar la herramienta
    herramienta_cct = create_retriever_tool(
        retriever,
        "buscar_contrato_colectivo",
        "Usa ESTA herramienta SIEMPRE para buscar sobre: vacaciones, aguinaldo, despidos, incapacidades, sindicato, permisos, horarios, salarios y cualquier derecho u obligación del IMSS. Si la pregunta es laboral, DEBES usarla."
    )
    tools = [herramienta_cct]

    # Mantenemos temperatura en 0.0 para evitar alucinaciones
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    agent_executor = create_react_agent(llm, tools)

    # --- 4) PROMPT DEL SISTEMA ANTI-ALUCINACIONES Y FLEXIBLE ---
    system_message = """Eres un asesor legal laboral experto en el marco normativo del IMSS (Contrato Colectivo de Trabajo, Reglamentos, Tabuladores, Profesiogramas, etc.).
Tu objetivo es ayudar a los trabajadores respondiendo sus dudas de forma clara, precisa y directa.

REGLAS ESTRICTAS E INQUEBRANTABLES (CERO ALUCINACIONES):
1. FUENTE EXCLUSIVA: Tu respuesta debe basarse ÚNICA Y EXCLUSIVAMENTE en el texto recuperado al usar la herramienta 'buscar_contrato_colectivo'. Tienes ESTRICTAMENTE PROHIBIDO usar tu conocimiento general o inventar información.
2. MANEJO DE VACÍOS (RESPUESTAS SEGURAS): 
   - Si la herramienta devuelve información que responde parcialmente a la pregunta, entrégala aclarando que es la única referencia encontrada en los documentos.
   - Si la herramienta NO devuelve ninguna información relacionada con la pregunta, NO INVENTES NI DEDUZCAS NADA. Responde exactamente: «No encontré la referencia exacta en los documentos cargados. ¿Podrías darme más detalles o usar el término técnico exacto de lo que buscas?».
3. CITAS PRECISAS Y ORIGEN: Siempre que fundamentes tu respuesta, debes especificar el número de cláusula/artículo y EL NOMBRE EXACTO DEL DOCUMENTO al que pertenece (si el fragmento lo indica).
   - Ejemplos correctos: "De acuerdo con el **Artículo X** del **Reglamento Interior de Trabajo**...", "Según la **Cláusula Y** del **Contrato Colectivo de Trabajo**...".
4. FORMATO Y PRESENTACIÓN:
   - Usa formato Markdown obligatoriamente.
   - Utiliza **negritas** para resaltar nombres de documentos, artículos, cláusulas, plazos y sanciones.
   - Usa viñetas (`- `) para listar requisitos, derechos o pasos.
   - Mantén un tono profesional, empático e institucional."""

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
