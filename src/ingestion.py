import os
import shutil
import pymupdf4llm
import pathlib
from tqdm import tqdm
import torch
from langdetect import detect, LangDetectException
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =============================================================================
# CONFIGURACIÓN: Rutas y Parámetros de Ingestión RAG 2.0
# =============================================================================
# El pipeline de ingestión está optimizado para:
# - Conversion PDF -> Markdown con pymupdf4llm (preserva estructura y tablas)
# - Chunking inteligente de 2000 caracteres (soporta contexto más amplio)
# - Embeddings nomic-ai/nomic-embed-text-v1.5 (multilingüe y robusto)
# - ChromaDB con Cosine Similarity y optimización HNSW
# - Soporte GPU/CUDA automático para acceleración

DATA_PATH = "data/pdfs/"
MD_DATA_PATH = "data/markdowns/"
CHROMA_PATH = "chromadb_storage"

def convert_pdfs_to_md():
    """
    Convierte PDF a Markdown usando pymupdf4llm.
    
    Ventajas:
    - Preserva estructura de títulos, listas y tablas
    - Mantiene relevancia semántica del documento
    - Reduce confusión en embeddings vs OCR puro
    """
    print("[1/4] Iniciando conversión de PDF a Markdown...")
    pathlib.Path(MD_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pathlib.Path(DATA_PATH).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ Advertencia: No se encontraron PDFs en {DATA_PATH}")
        return
    
    for pdf_path in pdf_files:
        md_file = pathlib.Path(MD_DATA_PATH) / f"{pdf_path.stem}.md"
        if not md_file.exists():
            try:
                print(f"  > Convirtiendo: {pdf_path.name}")
                md_text = pymupdf4llm.to_markdown(str(pdf_path))
                md_file.write_bytes(md_text.encode("utf-8"))
            except Exception as e:
                print(f"  ⚠️ Error al convertir {pdf_path.name}: {str(e)}")
        else:
            print(f"  ✓ {pdf_path.name} ya convertido")
    
    print(f"✅ Conversión completada. Documentos en: {MD_DATA_PATH}\n")

def load_md_documents():
    """
    Carga los archivos Markdown generados.
    
    Los documentos se cargan con metadatos de fuente para rastreo posterior
    en el pipeline de generación de rutinas.
    """
    print("[2/4] Cargando documentos Markdown...")
    try:
        loader = DirectoryLoader(
            MD_DATA_PATH,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        print(f"✅ Se cargaron {len(documents)} documentos Markdown\n")
        return documents
    except Exception as e:
        print(f"⚠️ Error al cargar documentos: {str(e)}")
        return []

def split_documents(documents):
    """
    Estrategia de Chunking 2.0 (Tarea 2 - Unidad 3):
    
    - chunk_size=2000: Aumentado de 1000 para aprovechar contexto más amplio
    - chunk_overlap=200: Mantiene coherencia entre fragmentos (10%)
    - Separadores jerárquicos: Respeta estructura lógica del documento
    
    Beneficios:
    - Nomic Embed soporta mejor contexto a 2000 caracteres
    - Reduce fragmentación de ideas importantes
    - Mejora relevancia en búsqueda semántica
    """
    print("[3/4] Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Aumentado de 1000 a 2000
        chunk_overlap=200,  # 10% de solapamiento
        separators=["\n\n", "\n", ".", " ", ""],  # Jerárquico
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Total de fragmentos generados: {len(chunks)}\n")
    return chunks


def detect_document_language(text):
    sample = (text or "").strip()[:4000]
    if len(sample) < 40:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def filter_english_documents(documents):
    print("Verificando idioma de Markdown y filtrando solo papers en inglés...")
    english_docs = []
    language_count = {}

    for doc in documents:
        language = detect_document_language(doc.page_content)
        doc.metadata["language"] = language
        language_count[language] = language_count.get(language, 0) + 1

        if language == "en":
            english_docs.append(doc)

    print("Conteo detectado por idioma:")
    for language, count in sorted(language_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {language}: {count}")

    print(f"Se conservaron {len(english_docs)} documentos en inglés de {len(documents)} totales.")
    return english_docs

def create_vector_store(chunks):
    """
    Crea base de datos vectorial con ChromaDB.
    
    Características:
    - Embeddings: nomic-ai/nomic-embed-text-v1.5 (768 dimensiones, multilingüe)
    - Metadatos: Se guarda fuente del documento en cada chunk
    - Distancia: Cosine Similarity (hnsw:space=cosine)
    - GPU: Soporte CUDA automático si está disponible
    - Batch Processing: Procesa en lotes de 50 para no saturar memoria
    
    Nota: ChromaDB se persiste en disco para futuras consultas sin recompilación.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/4] Creando base de datos vectorial...")
    print(f"  Dispositivo: {device.upper()}")
    print(f"  Total de fragmentos: {len(chunks)}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        batch_size = 50
        vector_db = None

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Generando embeddings",
            unit="batch",
        ):
            batch = chunks[i : i + batch_size]
            if vector_db is None:
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_PATH,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                vector_db.add_documents(batch)

        print(f"\n✅ Base de datos guardada en: {CHROMA_PATH}")
        print(f"  Total de vectores: {len(chunks)}\n")
        return vector_db

    except Exception as e:
        print(f"⚠️ Error al crear base de datos: {str(e)}")
        raise

if __name__ == "__main__":
    print("=" * 70)
    print("|           PIPELINE DE INGESTIÓN RAG 2.0 - UNIDAD 3          |")
    print("=" * 70)
    print()
    
    # Limpiar DB anterior para evitar conflictos dimensionales
    if os.path.exists(CHROMA_PATH):
        print(f"🔄 Limpiando base vectorial previa en: {CHROMA_PATH}")
        try:
            shutil.rmtree(CHROMA_PATH)
            print()
        except Exception as e:
            print(f"⚠️ Error al limpiar: {str(e)}\n")
    
    # Pipeline completo
    try:
        convert_pdfs_to_md()
        docs = load_md_documents()
        
        if not docs:
            print("⚠️ No se cargaron documentos. Abortando.")
            exit(1)
        
        english_docs = filter_english_documents(docs)
        
        if not english_docs:
            print("⚠️ No se encontraron documentos en inglés. Abortando.")
            exit(1)
        
        chunks = split_documents(english_docs)
        
        # Vista previa
        if len(chunks) > 0:
            print("Ejemplo del primer fragmento:")
            print("-" * 70)
            print(chunks[0].page_content[:400] + "...")
            print("-" * 70)
            print()
        
        db = create_vector_store(chunks)
        
        print("=" * 70)
        print("\u2705 INGESTIÓN COMPLETADA EXITOSAMENTE")
        print(f"   - Documentos procesados: {len(english_docs)}")
        print(f"   - Fragmentos creados: {len(chunks)}")
        print(f"   - Base de datos: {CHROMA_PATH}/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {str(e)}")
        exit(1)