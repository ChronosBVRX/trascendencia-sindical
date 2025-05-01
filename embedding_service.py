import openai
import json
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(texto):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return res.data[0].embedding

def load_embeddings():
    with open("fragmentos.json", "r", encoding="utf-8") as f:
        fragmentos = json.load(f)
    vectores = []
    for frag in fragmentos:
        embedding = get_embedding(frag["contenido"])
        vectores.append({
            "origen": frag["origen"],
            "contenido": frag["contenido"],
            "embedding": embedding
        })
    with open("vector_data.json", "w", encoding="utf-8") as f:
        json.dump(vectores, f, ensure_ascii=False)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def buscar_respuesta(pregunta):
    pregunta_embedding = get_embedding(pregunta)
    with open("vector_data.json", "r", encoding="utf-8") as f:
        vectores = json.load(f)

    mejor = max(vectores, key=lambda x: cosine_similarity(pregunta_embedding, x["embedding"]))
    contexto = f'Fuente: {mejor["origen"]}\nContenido:\n{mejor["contenido"][:2000]}'

    prompt = f"""Eres TrascendencIA Sindical, experto en contrato colectivo y reglamentos laborales del IMSS.
Basándote en el siguiente fragmento responde con lenguaje profesional y cálido. Cita el origen del contenido.

{contexto}

Pregunta: {pregunta}"""

    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.5
    )

    return respuesta.choices[0].message.content
