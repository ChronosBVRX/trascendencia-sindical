from langchain_core.documents import Document
import os
import fitz  # pymupdf

def cargar_contenido_pdfs(directorio="pdfs"):
    documentos = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(".pdf"):
            path = os.path.join(directorio, archivo)
            texto = extraer_texto_pdf(path)  # Usa aquí tu método para leer texto
            documentos.append(Document(page_content=texto, metadata={"fuente": archivo}))
    return documentos

def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with fitz.open(ruta_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def dividir_texto_en_fragmentos(texto, max_palabras=300):
    palabras = texto.split()
    fragmentos = []

    for i in range(0, len(palabras), max_palabras):
        fragmento = " ".join(palabras[i:i + max_palabras])
        fragmentos.append(fragmento)

    return fragmentos
