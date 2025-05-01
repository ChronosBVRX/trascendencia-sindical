import fitz  # PyMuPDF
import os
import json

def procesar_pdfs(carpeta_pdf):
    fragmentos = []
    for archivo in os.listdir(carpeta_pdf):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(carpeta_pdf, archivo)
            doc = fitz.open(ruta)
            for i, pagina in enumerate(doc):
                texto = pagina.get_text()
                if texto.strip():
                    fragmentos.append({
                        "contenido": texto,
                        "origen": f"{archivo} - página {i+1}"
                    })
    with open("fragmentos.json", "w", encoding="utf-8") as f:
        json.dump(fragmentos, f, ensure_ascii=False, indent=2)
