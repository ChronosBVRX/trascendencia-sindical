from pdf_loader import cargar_contenido_pdfs
from embedding_service import guardar_embeddings

def generar_y_guardar_vectorstore():
    print("📄 Cargando documentos PDF...")
    documentos = cargar_contenido_pdfs("pdfs")

    print(f"📦 Generando y guardando vectorstore con {len(documentos)} fragmentos...")
    guardar_embeddings(documentos, "vectorstore")
    print("✅ Base de datos vectorial generada correctamente.")

# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    generar_y_guardar_vectorstore()
