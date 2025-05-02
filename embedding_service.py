```python
# embedding_service.py
import os
import fitz
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
