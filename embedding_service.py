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


def consulta_contrato(question: str, history: List[dict]) -> str:
    """
    1) Corrige ortografía de la pregunta
    2) Consulta FAISS sobre la versión corregida
    3) Genera respuesta creativa y natural, citando sección y artículo
    """
    # 1) CORRECCIÓN ORTOGRÁFICA SILENCIOSA
    ortho_model = ChatOpenAI(temperature=0)
    ortho_msgs = [
        SystemMessage(content=(
            "Eres un corrector ortográfico: recibe una pregunta y devuelve solo la versión "
            "con ortografía y gramática corregidas, sin añadir ni quitar nada del contenido."
        )),
        HumanMessage(content=question)
    ]
    question_corr = ortho_model(ortho_msgs).content.strip()

    # 2) CARGA INDICE Y BÚSQUEDA
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    top_docs = db.similarity_search(question_corr, k=5)
    contexto = "\n".join(f"— Fragmento:\n{d.page_content}" for d in top_docs)

    # 3) PROMPT PARA LA RESPUESTA FINAL
    system_prompt = """
Eres un asistente experto en el Contrato Colectivo de Trabajo del IMSS.
Habla de forma creativa y natural, como si conversarás con un amigo.  
Al responder:
1) Menciona la sección exacta (ej. "Reglamento Interior de Trabajo").  
2) Indica el número de artículo o cláusula.  
3) Extrae **literalmente** el texto relevante.  
4) No inventes ni alucines referencias; si no lo encuentras, di:
   «No se encontró referencia exacta en el contrato.»  
5) Ten en cuenta que el usuario puede haber fallado en la ortografía; interpreta la pregunta tras corregirla internamente.
""".strip()

    # Reconstruimos el hilo de la conversación
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Añadimos el contexto y la pregunta corregida
    messages.append(HumanMessage(content=(
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta (corregida): {question_corr}"
    )))

    chat = ChatOpenAI(temperature=0.4)
    return chat(messages).content
