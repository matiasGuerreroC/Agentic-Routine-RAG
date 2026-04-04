import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Librería moderna para embeddings
from langchain_community.vectorstores import Chroma

# Configuración de rutas
DATA_PATH = "data/"
CHROMA_PATH = "chromadb_storage"

def load_documents():
    print("Cargando documentos...")
    # Cargamos todos los PDFs de la carpeta data
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Se cargaron {len(documents)} páginas en total.")
    return documents

def split_documents(documents):
    # El splitter divide el texto en pedazos manejables
    # chunk_size: tamaño del trozo (en caracteres)
    # chunk_overlap: cuánto se traslapa cada trozo con el anterior (para no perder contexto)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} trozos.")
    return chunks

def create_vector_store(chunks):
    print("Iniciando creación de base de datos vectorial...")
    
    # 1. Definir el modelo de embeddings (all-MiniLM-L6-v2 es rápido y eficiente)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Crear y persistir la base de datos en la carpeta especificada
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Base de datos guardada exitosamente en: {CHROMA_PATH}")
    return vector_db

if __name__ == "__main__":
    # Prueba inicial de carga
    docs = load_documents()
    chunks = split_documents(docs)
    
    # Ver un ejemplo de un trozo
    if len(chunks) > 0:
        print("\nEjemplo del primer trozo:")
        print(chunks[0].page_content[:200] + "...")
        
    db = create_vector_store(chunks)
    
    print("----------------------------------------")
    print("Proceso de ingestión completado.")
    print("----------------------------------------")