import os
from fastapi import FastAPI
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

# 1) Carga variables de entorno (.env) con OPENAI_API_KEY
load_dotenv()

# 2) Crea la app de FastAPI
app = FastAPI()

# 3) Define las rutas ABSOLUTAS a los artefactos FAISS
BASE = os.getcwd()
VECTORSTORE_PATH = os.path.join(BASE, "vectorstore", "index.faiss")
PICKLE_PATH      = os.path.join(BASE, "vectorstore", "index.pkl")

@app.on_event("startup")
def startup_event():
    """
    Al iniciar, si falta el índice FAISS o el pickle,
    lo genera leyendo los PDFs de /pdfs.
    """
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        print("🔄 Vectorstore no encontrado. Generando índice desde PDFs...")
        try:
            generar_y_guardar_vectorstore()
            print("✅ Índice FAISS generado correctamente.")
        except Exception as e:
            print("❌ Error generando índice FAISS:", e)

@app.post("/consulta")
def endpoint_consulta(payload: dict):
    """
    Recibe POST { "texto": "tu pregunta" }
    Devuelve   { "respuesta": "…tu respuesta…" }
    """
    pregunta = payload.get("texto", "").strip()
    if not pregunta:
        return {"respuesta": "❗ Envía tu pregunta en el campo 'texto'."}

    try:
        respuesta = consulta_contrato(pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"Error interno procesando la consulta: {e}"}
