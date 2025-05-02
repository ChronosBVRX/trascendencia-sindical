from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_loader import cargar_contenido_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_service import guardar_embeddings, cargar_documentos

def generar_y_guardar_vectorstore():
    documentos = cargar_documentos("pdfs")  # o el nombre de tu carpeta
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos_divididos = splitter.split_documents(documentos)
    
    print(f"Se dividieron {len(documentos)} documentos en {len(documentos_divididos)} fragmentos.")
    guardar_embeddings(documentos_divididos, "vectorstore")

def dividir_documentos(documentos):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_documents(documentos)

def generar_y_guardar_vectorstore():
    documentos = cargar_documentos()  # o cargar_tus_docs(), etc.
    documentos_divididos = dividir_documentos(documentos)
    guardar_embeddings(documentos_divididos, "vectorstore")

def generar_y_guardar_vectorstore():
    documentos = cargar_contenido_pdfs("pdfs")
    guardar_embeddings(documentos, "vectorstore")
    print(f"✅ Base de datos vectorial creada con {len(documentos)} fragmentos.")
