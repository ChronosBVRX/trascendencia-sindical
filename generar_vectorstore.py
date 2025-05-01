from pdf_loader import cargar_contenido_pdfs
from embedding_service import guardar_embeddings

# Paso 1: cargar los textos desde la carpeta de PDFs
documentos = cargar_contenido_pdfs("pdfs")

# Paso 2: generar y guardar embeddings
guardar_embeddings(documentos, "vectorstore")

print(f"✅ Base de datos vectorial creada con {len(documentos)} fragmentos.")
