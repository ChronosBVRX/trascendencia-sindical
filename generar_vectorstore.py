def generar_y_guardar_vectorstore():
    from pdf_loader import cargar_contenido_pdfs
    from embedding_service import guardar_embeddings

    print("📦 Generando base de datos vectorial...")
    documentos = cargar_contenido_pdfs("pdfs")
    guardar_embeddings(documentos, "vectorstore")
    print(f"✅ Vectorstore creado con {len(documentos)} fragmentos.")
