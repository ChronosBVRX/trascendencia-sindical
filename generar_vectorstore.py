from pdf_loader import cargar_contenido_pdfs
from embedding_service import guardar_embeddings

def generar_y_guardar_vectorstore():
    documentos = cargar_contenido_pdfs("pdfs")
    guardar_embeddings(documentos, "vectorstore")
    print(f"✅ Base de datos vectorial creada con {len(documentos)} fragmentos.")
