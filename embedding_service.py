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
def guardar_embeddings(documentos, path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documentos, embeddings)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/index.faiss", "wb") as f1, open(f"{path}/index.pkl", "wb") as f2:
        pickle.dump(vectorstore.index, f1)
        pickle.dump(vectorstore, f2)

    print(f"✅ Vectorstore guardado en {path}")

# Para buscar respuesta basada en una pregunta
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
