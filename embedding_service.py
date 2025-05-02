import os
import openai
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Para cargar embeddings desde disco
def load_embeddings(path="vectorstore"):
    with open(f"{path}/index.faiss", "rb") as f1, open(f"{path}/index.pkl", "rb") as f2:
        index = pickle.load(f1)
        vectorstore = pickle.load(f2)
        vectorstore.index = index
        return vectorstore

# Para guardar embeddings desde documentos
def guardar_embeddings(documentos, path):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documentos, embeddings)
    
    embeddings = OpenAIEmbeddings()
    texts = [d.page_content for d in documentos]
    metadatas = [d.metadata for d in documentos]

    # Lote de procesamiento (por ejemplo, 100 documentos por vez)
    chunk_size = 100
    vectorstore = None

    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_metadatas = metadatas[i:i + chunk_size]
        partial_vs = FAISS.from_texts(chunk_texts, embeddings, metadatas=chunk_metadatas)
        if vectorstore is None:
            vectorstore = partial_vs
        else:
            vectorstore.merge_from(partial_vs)

    vectorstore.save_local(path)

def buscar_respuesta(pregunta, vectorstore):
    documentos_relacionados = vectorstore.similarity_search(pregunta, k=4)
    contexto = "\n\n".join([doc.page_content for doc in documentos_relacionados])

    prompt = f"""
Eres TrascendencIA Sindical, un asistente experto en el Contrato Colectivo de Trabajo del IMSS.

Tu trabajo es responder **únicamente con base en el contenido textual del contrato** incluido a continuación. 
No inventes cláusulas ni interpretes. Si no puedes citar una cláusula, responde que no la encontraste.

=== CONTENIDO DEL CONTRATO ===
{contexto}

=== PREGUNTA ===
{pregunta}

=== RESPUESTA ===
Cita textualmente si es posible y di en qué reglamento o cláusula se encuentra. Si no puedes encontrarlo, indícalo con claridad.
"""

    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return respuesta.choices[0].message.content.strip()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Para guardar embeddings desde documentos
def guardar_embeddings(documentos, path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documentos, embeddings)
    vectorstore.save_local(path)
    print(f"✅ Embeddings guardados en '{path}'")
