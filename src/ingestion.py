import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configurar la ruta de los datos
DATA_PATH = "data/"

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

if __name__ == "__main__":
    # Prueba inicial de carga
    docs = load_documents()
    chunks = split_documents(docs)
    
    # Ver un ejemplo de un trozo
    if len(chunks) > 0:
        print("\nEjemplo del primer trozo:")
        print(chunks[0].page_content[:200] + "...")