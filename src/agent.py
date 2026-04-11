import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar configuración
load_dotenv()
CHROMA_PATH = "chromadb_storage"

# 2. Configurar LLMs (Usamos 2 temperaturas distintas para Self-Consistency)
# LLM Generador: Temperatura 0.4 para tener variedad en las respuestas
llm_generator = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.4,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# LLM Juez: Temperatura 0.1 para que sea estricto al elegir la mejor opción
llm_judge = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 3. Configurar el Recuperador (RAG)
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join([f"Fuente: {d.metadata.get('source')}\nContenido: {d.page_content}" for d in docs])

# 4. PROMPT AVANZADO: Few-Shot + Chain-of-Thought (CoT)
template_cot = """
Actúa como un entrenador personal experto basado estrictamente en evidencia científica deportiva.
Debes diseñar una rutina utilizando EXCLUSIVAMENTE el contexto proporcionado y el equipamiento disponible.

REGLAS DE RAZONAMIENTO (Chain-of-Thought):
Antes de dar la rutina, haz un análisis explícito:
1. ¿Qué equipo tiene disponible el usuario?
2. ¿Tiene alguna lesión o dolor mencionado? (Si hay dolor, evita ejercicios que afecten esa zona).
3. ¿Qué volumen recomienda la evidencia recuperada?

--- EJEMPLO DE FORMATO ESPERADO ---
PREGUNTA: "Tengo bandas elásticas, dolor de hombro. Quiero rutina de tren superior 2 días."
CONTEXTO: "Se recomiendan 3 series de 10 reps. Ejercicios: flexiones, remo con banda."

ANÁLISIS:
- Equipo: Bandas elásticas.
- Lesión: Dolor de hombro (Evitar flexiones y ejercicios de empuje directo).
- Volumen: 3 series de 10 reps.

**1. Parámetros de Entrenamiento:**
- Series: 3
- Repeticiones: 10

**2. Rutina Propuesta:**
- Remo con banda (espalda)
(Omitimos flexiones por precaución al hombro)

**3. Justificación Científica:**
Basado en la evidencia, se aplica tensión mecánica con bandas. Se excluyen empujes por dolor articular.
--------------------------------------------------

CONTEXTO CIENTÍFICO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: 
{question}

TU RESPUESTA (Incluye ANÁLISIS seguido de las 3 secciones):
"""

prompt_cot = ChatPromptTemplate.from_template(template_cot)

# 5. Pipeline RAG Generador
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_cot
    | llm_generator
    | StrOutputParser()
)

# 6. Limpiador de formato (Elimina el tag <think> de Qwen)
def clean_qwen_output(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# 7. FUNCIÓN SELF-CONSISTENCY (El Orquestador)
def generar_rutina_robusta(pregunta_usuario):
    print("⏳ [Paso 1] Generando múltiples caminos de razonamiento (Self-Consistency)...")
    respuestas =[]
    
    # Generamos 3 rutinas posibles
    for i in range(3):
        print(f"   Generando opción {i+1}...")
        raw_res = rag_chain.invoke(pregunta_usuario)
        respuestas.append(clean_qwen_output(raw_res))
        
    print("\n⚖️ [Paso 2] Evaluando la opción más segura y consistente (LLM Juez)...")
    
    # El Juez evalúa cuál de las 3 rutinas es mejor
    judge_template = """
    Eres un supervisor clínico-deportivo. Revisa estas 3 opciones de rutina generadas para la siguiente petición:
    PREGUNTA DEL USUARIO: "{question}"
    
    Evalúa estrictamente:
    1. ¿Respeta exactamente el equipamiento disponible?
    2. ¿Es la más segura respecto a los dolores mencionados?
    3. ¿Tiene formato Markdown limpio?
    
    OPCIÓN 1:
    {op1}
    ---
    OPCIÓN 2:
    {op2}
    ---
    OPCIÓN 3:
    {op3}
    
    Devuelve ÚNICAMENTE el texto completo de la MEJOR OPCIÓN, sin agregar comentarios tuyos al principio ni al final.
    """
    
    judge_prompt = ChatPromptTemplate.from_template(judge_template)
    judge_chain = judge_prompt | llm_judge | StrOutputParser()
    
    mejor_rutina_raw = judge_chain.invoke({
        "question": pregunta_usuario,
        "op1": respuestas[0],
        "op2": respuestas[1],
        "op3": respuestas[2]
    })
    
    return clean_qwen_output(mejor_rutina_raw)

# --- EJECUCIÓN (DEMO EN VIVO) ---
if __name__ == "__main__":
    # La pregunta de prueba ("Stress Test")
    pregunta = """
    Hola, soy principiante y quiero aumentar la masa muscular de mi tren superior y piernas 
    entrenando en mi casa 3 veces por semana. Estrictamente NO tengo pesas ni mancuernas, 
    solo puedo usar mi peso corporal (calistenia) y una silla. 
    Además, me duele un poco la rodilla derecha al flectarla mucho. 
    ¿Me puedes armar una rutina y explicarme cuántas series y descansos necesito según la ciencia?
    """
    
    print("🚀 INICIANDO PIPELINE DE AGENTES AVANZADO\n")
    rutina_final = generar_rutina_robusta(pregunta)
    
    print("\n" + "="*60)
    print(" ✅ RUTINA FINAL APROBADA (Formato Markdown Robusto) ")
    print("="*60 + "\n")
    print(rutina_final)