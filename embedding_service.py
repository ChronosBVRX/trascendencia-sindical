# embedding_service.py

import os
import openai
import pickle
from langchain.vectorstores import FAISS

# Para cargar embeddings desde disco
def load_embeddings(path="vectorstore/index"):
    with open(f"{path}/index.faiss", "rb") as f1, open(f"{path}/index.pkl", "rb") as f2:
        index = pickle.load(f2)
        faiss_obj = FAISS.load_local(path, index)
        return faiss_obj

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