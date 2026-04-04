import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar la API KEY desde el .env
load_dotenv()

CHROMA_PATH = "chromadb_storage"

# 2. Configurar el LLM con tu modelo específico
llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 3. Configurar el modelo de Embeddings Multilingüe
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# 4. Cargar la base de datos vectorial
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 5. Configurar el recuperador (traer los 5 fragmentos más relevantes)
retriever = db.as_retriever(search_kwargs={"k": 5})

# 6. Definir el System Prompt
template = """
Actúa como un entrenador personal experto basado en evidencia científica deportiva. 
Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE el contexto científico proporcionado.

Contexto de los papers:
{context}

Pregunta del usuario: 
{question}

Respuesta:
"""

prompt = ChatPromptTemplate.from_template(template)

# 7. Formatear los documentos encontrados
def format_docs(docs):
    return "\n\n".join([f"Fuente: {d.metadata.get('source')}\nContenido: {d.page_content}" for d in docs])

# 8. Crear la cadena RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # Prueba de concepto
    pregunta = "¿Qué dicen los manuales sobre los beneficios de la hipertrofia y la sobrecarga progresiva?"
    
    print(f"🚀 Consultando al experto deportivo con Qwen...\n")
    try:
        resultado = rag_chain.invoke(pregunta)
        print("--- RESPUESTA DEL AGENTE ---")
        print(resultado)
    except Exception as e:
        print(f"Hubo un error con el modelo: {e}")
        print("Nota: Si dice 'model not found', prueba con 'qwen-2.5-32b' que es el ID estándar de Groq.")