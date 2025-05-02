import os
from fastapi import FastAPI
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

# 1) Carga la clave de OpenAI desde .env
load_dotenv()

# 2) Crea la app de FastAPI
app = FastAPI()

# 3) Define dónde buscar los archivos generados
BASE = os.getcwd()
VECTORSTORE_PATH = os.path.join(BASE, "vectorstore", "index.faiss")
PICKLE_PATH      = os.path.join(BASE, "vectorstore", "index.pkl")

@app.on_event("startup")
def startup_event():
    """
    Al iniciar, comprueba si el vectorstore existe.
    Si falta, lo genera leyendo los PDFs en /pdfs.
    """
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        print("🔄 Vectorstore no encontrado. Generando índice desde PDFs...")
        try:
            generar_y_guardar_vectorstore()
            print("✅ Índice generado correctamente.")
        except Exception as e:
            print("❌ Error generando índice:", e)

@app.post("/consulta")
def endpoint_consulta(payload: dict):
    """
    Recibe POST con JSON {"texto": "..."} y devuelve {"respuesta": "..."}.
    Maneja preguntas vacías y errores internos.
    """
    pregunta = payload.get("texto", "").strip()
    if not pregunta:
        return {"respuesta": "❗ Por favor envía tu pregunta en el campo 'texto'."}

    try:
        respuesta = consulta_contrato(pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        # Si algo falla dentro de la consulta, lo reportamos
        return {"error": f"Error interno al procesar la consulta: {e}"}
