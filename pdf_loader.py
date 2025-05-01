from langchain_core.documents import Document
import os
import fitz  # pymupdf

def cargar_contenido_pdfs(carpeta_pdf):
    documentos = []

    for archivo in os.listdir(carpeta_pdf):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(carpeta_pdf, archivo)
            texto = extraer_texto_pdf(ruta)
            fragmentos = dividir_texto_en_fragmentos(texto, max_palabras=300)

            for frag in fragmentos:
                documentos.append({
                    "contenido": frag,
                    "origen": archivo
                })

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
