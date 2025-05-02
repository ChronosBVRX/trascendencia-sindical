import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from embedding_service import generar_y_guardar_vectorstore, consulta_contrato

# 1) Carga tu .env
load_dotenv()
app = FastAPI()

# 2) Montamos ONLY /static para ficheros estáticos
BASE = os.getcwd()
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE, "static")),
    name="static"
)

# 3) GET / → devuelve index.html
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

# 4) Nuestro POST /consulta
@app.post("/consulta")
async def endpoint_consulta(payload: dict):
    pregunta = (payload.get("texto") or "").strip()
    if not pregunta:
        return {"respuesta": "❗ Envía tu pregunta en el campo 'texto'."}
    try:
        return {"respuesta": consulta_contrato(pregunta)}
    except Exception as e:
        return {"error": f"Error interno: {e}"}

# 5) Startup: construye el vectorstore si no existe
VECTORSTORE_DIR  = os.path.join(BASE, "vectorstore")
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
PICKLE_PATH      = os.path.join(VECTORSTORE_DIR, "index.pkl")

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(PICKLE_PATH):
        generar_y_guardar_vectorstore()
