import os
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()
openai = OpenAI()

# Carga y decodificación del texto
def generar_embedding(texto):
    response = openai.embeddings.create(
        input=[texto],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def guardar_embeddings(documentos, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    for i, doc in enumerate(documentos):
        embedding = generar_embedding(doc["contenido"])
        with open(os.path.join(carpeta_destino, f"vector_{i}.pkl"), "wb") as f:
            pickle.dump({
                "embedding": embedding,
                "contenido": doc["contenido"],
                "origen": doc["origen"]
            }, f)

def load_embeddings(carpeta_origen):
    vectores = []
    for archivo in os.listdir(carpeta_origen):
        if archivo.endswith(".pkl"):
            with open(os.path.join(carpeta_origen, archivo), "rb") as f:
                vectores.append(pickle.load(f))
    return vectores

def buscar_respuesta(pregunta, embeddings, k=3):
    pregunta_emb = generar_embedding(pregunta)
    similitudes = []

    for doc in embeddings:
        distancia = np.dot(pregunta_emb, doc["embedding"])
        similitudes.append((distancia, doc))

    similitudes.sort(reverse=True, key=lambda x: x[0])
    mejores = similitudes[:k]

    contexto = "\n\n".join([m[1]["contenido"] for m in mejores])

    respuesta = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asesor sindical experto en el contrato colectivo del IMSS. Responde de forma profesional y amable, citando siempre que sea posible el contenido original del contrato."},
            {"role": "user", "content": f"Con base en el siguiente contexto:\n\n{contexto}\n\nResponde a esta pregunta:\n{pregunta}"}
        ]
    )

    return respuesta.choices[0].message.content.strip()
