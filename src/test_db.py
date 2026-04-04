from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # <--- Librería actualizada

CHROMA_PATH = "chromadb_storage"

def test_query(query_text):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Cargamos la DB con la nueva librería
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    print(f"\n--- Buscando información para: '{query_text}' ---")
    
    # Usamos similarity_search_with_score (devuelve DISTANCIA)
    # k=3 para traer los 3 más cercanos
    results = db.similarity_search_with_score(query_text, k=3)
    
    for i, (doc, score) in enumerate(results):
        # En distancia: 0 es idéntico, 1+ es muy diferente
        print(f"\n--- Resultado {i+1} (Distancia: {score:.4f}) ---")
        print(f"Fuente: {doc.metadata.get('source', 'Desconocida')}")
        print(f"Contenido: {doc.page_content[:400]}...")

if __name__ == "__main__":
    test_query("¿Cómo se aplica la sobrecarga progresiva en casa?")