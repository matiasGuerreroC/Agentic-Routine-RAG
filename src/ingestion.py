import os
import shutil
import pymupdf4llm
import pathlib
from tqdm import tqdm
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuración de rutas
DATA_PATH = "data/pdfs/"
MD_DATA_PATH = "data/markdowns/"
CHROMA_PATH = "chromadb_storage"

def convert_pdfs_to_md():
    """Convierte todos los PDFs en la carpeta data a archivos Markdown."""
    print("Iniciando conversión de PDF a Markdown...")
    pathlib.Path(MD_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pathlib.Path(DATA_PATH).glob("*.pdf"):
        md_file = pathlib.Path(MD_DATA_PATH) / f"{pdf_path.stem}.md"
        if not md_file.exists():
            print(f"  > Convirtiendo: {pdf_path.name}")
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            md_file.write_bytes(md_text.encode("utf-8"))
    print("Conversión completada.\n")

def load_md_documents():
    """Carga los archivos Markdown generados."""
    print("Cargando documentos Markdown...")
    loader = DirectoryLoader(MD_DATA_PATH, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    print(f"Se cargaron {len(documents)} documentos Markdown.")
    return documents

def split_documents(documents):
    """
    Estrategia de Chunking 2.0:
    Como BGE-M3 soporta mucho contexto, subimos el chunk_size.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # Aumentado de 1000 a 2000
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} trozos.")
    return chunks

def create_vector_store(chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    print(f"Iniciando creación de base de datos vectorial con {len(chunks)} trozos...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Procesar en lotes para no saturar la memoria y ver progreso
    batch_size = 50 
    vector_db = None
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generando Embeddings"):
        batch = chunks[i : i + batch_size]
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vector_db.add_documents(batch)
            
    print(f"\n✅ Base de datos guardada exitosamente en: {CHROMA_PATH}")
    return vector_db

if __name__ == "__main__":
    # Recrear la DB evita choques de dimensión cuando se cambia de modelo.
    if os.path.exists(CHROMA_PATH):
        print(f"Limpiando base vectorial previa en: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    convert_pdfs_to_md()
    docs = load_md_documents()
    chunks = split_documents(docs)
    
    if len(chunks) > 0:
        print(f"\nEjemplo del primer trozo (Markdown):\n{chunks[0].page_content[:300]}...")
        
    db = create_vector_store(chunks)
    print("\n✅ Proceso de ingestión 2.0 completado.")