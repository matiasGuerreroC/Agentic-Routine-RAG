import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- SETUP INICIAL ---
llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.1, max_tokens=4000, groq_api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="chromadb_storage", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join([f"Fuente: {d.metadata.get('source')}\nContenido: {d.page_content[:800]}" for d in docs])

def clean_qwen_output(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# --- PREGUNTA STRESS TEST ---
question = """
Hola, soy principiante y quiero aumentar la masa muscular de mi tren superior y piernas 
entrenando en mi casa 3 veces por semana. Estrictamente NO tengo pesas ni mancuernas, 
solo puedo usar mi peso corporal (calistenia) y una silla. 
Además, me duele un poco la rodilla derecha al flectarla mucho. 
¿Me puedes armar una rutina y explicarme cuántas series y descansos necesito según la ciencia?
"""

# ==========================================
# ESTRATEGIA 1: ZERO-SHOT
# ==========================================
template_zs = """
Actúa como un entrenador personal experto basado estrictamente en evidencia científica deportiva.
Tu objetivo es diseñar rutinas de entrenamiento utilizando EXCLUSIVAMENTE el contexto proporcionado.

Reglas estrictas:
1. Usa solo el equipamiento que tiene el usuario.
2. Estructura en 3 partes: Volumen, Rutina, Justificación.
3. No des consejos médicos. Si hay dolor, da una advertencia general.

CONTEXTO: {context}
PREGUNTA: {question}
RESPUESTA:
"""
prompt_zs = ChatPromptTemplate.from_template(template_zs)
chain_zs = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_zs | llm | StrOutputParser()


# ==========================================
# ESTRATEGIA 2: FEW-SHOT
# ==========================================
template_fs = """
Actúa como un entrenador personal experto basado estrictamente en evidencia científica deportiva.
Tu objetivo es diseñar rutinas de entrenamiento utilizando EXCLUSIVAMENTE el contexto proporcionado.

Reglas estrictas de comportamiento:
1. Usa solo el equipamiento que tiene el usuario. Si pide algo que no está en el contexto, indícalo amablemente.
2. Si el contexto indica un número de series, repeticiones o descansos, úsalos exactamente.
3. REGLA DE SALUD: Si el usuario menciona dolor o molestias, INICIA tu respuesta con un bloque "**⚠️ ADVERTENCIA DE SEGURIDAD:**" recomendando consultar a un médico, y adapta la rutina omitiendo ejercicios que involucren esa zona. Si NO hay dolor, omite esta advertencia.
4. ESTRUCTURA DE LA RUTINA: 
   - Si el usuario menciona una cantidad de días a la semana, agrupa los ejercicios por día (ej. **Día 1: Enfoque...**).
   - Si NO menciona días, haz una lista secuencial de los ejercicios (1, 2, 3...).
   - En AMBOS casos, CADA ejercicio debe incluir obligatoriamente 3 viñetas: *Músculos objetivos*, *Cómo hacerlo* y *Modificación* (Agregar solo por dolor o falta de equipo).
5. Estructura tu respuesta final en 3 partes claras: 1. Parámetros de Entrenamiento, 2. Rutina Propuesta, 3. Justificación Científica.

--- EJEMPLO 1 (Menciona DÍAS y SÍ tiene DOLOR) ---
Pregunta: "Quiero entrenar tren superior 2 días a la semana en casa. Tengo bandas elásticas y dolor de codo."
Contexto: "Hipertrofia con bandas: 3-4 series, 10-15 reps. Descanso 60s. Ejercicios: Remo con banda, Aperturas de pecho, Press (evitar con dolor articular en codos)."
Tu Respuesta:
**⚠️ ADVERTENCIA DE SEGURIDAD:** Como entrenador, te recomiendo que si el dolor de codo es persistente, suspendas el ejercicio y consultes a un médico. Esta rutina omite movimientos de empuje directo por precaución.

### **1. Parámetros de Entrenamiento:**
- **Series:** 3 a 4 por ejercicio.
- **Repeticiones:** 10 a 15.
- **Descanso:** 60 segundos.

### **2. Rutina Propuesta:**
**Día 1: Tren superior tracción**
1. **Remo con banda**
   - *Músculos objetivos:* Dorsales y bíceps.
   - *Cómo hacerlo:* Sentado en el suelo, pasar la banda por los pies y tirar hacia el ombligo.
   - *Modificación:* Mantener los codos pegados al cuerpo para no forzar la articulación.

**Día 2: Tren superior aislamiento**
1. **Aperturas de pecho con banda (Chest flys)**
   - *Músculos objetivos:* Pectorales.
   - *Cómo hacerlo:* De pie, anclar la banda en la espalda y juntar los brazos extendidos al frente.
   - *Modificación:* Rango de movimiento corto, manteniendo una leve flexión estática del codo.

### **3. Justificación Científica:**
Se excluyeron ejercicios de empuje multiarticulares para proteger el codo, respetando los volúmenes de hipertrofia del contexto.

--- EJEMPLO 2 (NO menciona días y SÍ tiene DOLOR) ---
Pregunta: "Quiero hacer piernas con peso corporal y una silla. Me duele la espalda baja al agacharme."
Contexto: "Ejercicios peso corporal: Zancadas, Puente de glúteo, Sentadillas (evitar con dolor lumbar). Volumen: 3 series de 12 reps."
Tu Respuesta:
**⚠️ ADVERTENCIA DE SEGURIDAD:** Si presentas dolor en la espalda baja, te aconsejo consultar con un profesional de la salud. La siguiente rutina omite flexiones profundas de tronco por precaución.

### **1. Parámetros de Entrenamiento:**
- **Series:** 3
- **Repeticiones:** 12
- **Descanso:** 60 a 90 segundos.

### **2. Rutina Propuesta:**
1. **Zancadas estáticas**
   - *Músculos objetivos:* Cuádriceps y glúteos.
   - *Cómo hacerlo:* Dar un paso largo y bajar la rodilla trasera hacia el suelo manteniendo la espalda recta.
   - *Modificación:* Usar el respaldo de la silla como apoyo para mantener el equilibrio y el torso recto.

### **3. Justificación Científica:**
Se seleccionaron ejercicios que minimizan la carga axial en la columna lumbar según la evidencia.
--------------------------------------------------

CONTEXTO CIENTÍFICO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: 
{question}

TU RESPUESTA:
"""
prompt_fs = ChatPromptTemplate.from_template(template_fs)
chain_fs = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_fs | llm | StrOutputParser()


# ==========================================
# ESTRATEGIA 3: CHAIN-OF-THOUGHT (CoT) CON GUARDRAILS
# ==========================================
template_cot = """
Actúa como un entrenador personal experto.
Antes de generar la rutina final, debes realizar un análisis interno de RESTRICCIONES (Chain-of-Thought).

Sigue este proceso de razonamiento EXACTO y muéstralo en tu respuesta:
1. [Análisis de Equipo]: ¿Qué tiene disponible y qué está prohibido?
2. [Análisis de Salud]: ¿Hay dolor? Si es así, ¿qué ejercicios del contexto debo omitir por precaución?
3. [Análisis de Días]: ¿El usuario mencionó una cantidad de días a la semana? (Esto determina si la rutina se agrupa por "Días" o es una lista secuencial).
4.[Métricas RAG]: ¿Qué volumen/descanso dice el contexto?
5. [Generación]: Crear la rutina final. Cada ejercicio DEBE incluir las viñetas: *Músculos objetivos*, *Cómo hacerlo* y *Modificación* (Agregar solo por dolor o falta de equipo). Si hubo dolor, incluye la ⚠️ ADVERTENCIA DE SEGURIDAD al inicio.

--- EJEMPLO DE RAZONAMIENTO (Few-Shot CoT) ---
PREGUNTA: "Tengo bandas elásticas, dolor de hombro. Quiero rutina de 2 días."
CONTEXTO: "3 series de 10 reps. Ejercicios: flexiones, remo con banda."

RAZONAMIENTO PASO A PASO Y RESPUESTA:
1. [Análisis de Equipo]: Tiene bandas. Prohibidas pesas.
2.[Análisis de Salud]: Dolor de hombro. Omitir flexiones.
3. [Análisis de Días]: Sí, mencionó 2 días. La rutina debe ir agrupada por "Día 1" y "Día 2".
4.[Métricas RAG]: 3 series de 10 repeticiones.
5. [Generación]:
**⚠️ ADVERTENCIA DE SEGURIDAD:** Por tu dolor de hombro, te sugiero consultar a un médico. He omitido ejercicios de empuje directo.

### **1. Parámetros de Entrenamiento:**
- **Series:** 3 a 4 por ejercicio.
- **Repeticiones:** 10 a 15.
- **Descanso:** 60 segundos.

### **2. Rutina Propuesta:**
**Día 1: Tren superior tracción**
1. **Remo con banda**
   - *Músculos objetivos:* Dorsales y bíceps.
   - *Cómo hacerlo:* Sentado en el suelo, pasar la banda por los pies y tirar hacia el ombligo.
   - *Modificación:* Mantener los codos pegados al cuerpo para no forzar la articulación.

**Día 2: Tren superior aislamiento**
1. **Aperturas de pecho con banda (Chest flys)**
   - *Músculos objetivos:* Pectorales.
   - *Cómo hacerlo:* De pie, anclar la banda en la espalda y juntar los brazos extendidos al frente.
   - *Modificación:* Rango de movimiento corto, manteniendo una leve flexión estática del codo.

### **3. Justificación Científica:**
Se excluyeron ejercicios de empuje multiarticulares para proteger el codo, respetando los volúmenes de hipertrofia del contexto.
--------------------------------------------------

CONTEXTO CIENTÍFICO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: 
{question}

RAZONAMIENTO PASO A PASO Y RESPUESTA:
"""
prompt_cot = ChatPromptTemplate.from_template(template_cot)
chain_cot = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_cot | llm | StrOutputParser()

# ==========================================
# ESTRATEGIA 4: SELF-CONSISTENCY (Versión Robusta)
# ==========================================
def run_self_consistency(pregunta):
    print("\n" + "="*50)
    print("EJECUTANDO SELF-CONSISTENCY (Votación de 3 rutas)...")
    print("="*50)
    
    respuestas = []
    # Usamos el generador Qwen
    llm_variant = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.5, max_tokens=2000, groq_api_key=os.getenv("GROQ_API_KEY"))
    chain_variant = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_cot | llm_variant | StrOutputParser()
    
    for i in range(3):
        print(f"   Generando opción de razonamiento {i+1}...")
        try:
            res = clean_qwen_output(chain_variant.invoke(pregunta))
            respuestas.append(res)
            # PAUSA CRÍTICA: Esperamos 20 segundos para que Groq reinicie el contador de tokens
            if i < 2: 
                print("   ⏳ Esperando 20s para evitar Rate Limit (TPM)...")
                time.sleep(20) 
        except Exception as e:
            print(f"   ⚠️ Error en muestra {i+1}: {e}")
            continue
    
    if len(respuestas) < 2:
        return "Error: No se pudieron generar suficientes muestras para consistencia."

    print("\nEl Juez (Llama 3.3 70B) está evaluando la mejor opción...")
    
    # El Juez necesita un modelo con MAS LÍMITE de tokens (Llama 3.3 tiene 12k TPM)
    llm_judge = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=os.getenv("GROQ_API_KEY"))
    
    judge_template = """
    Actúa como un Auditor de Seguridad Deportiva. Revisa estas 3 propuestas de rutina para un usuario que dijo:
    "{question}"
    
    Criterios de auditoría (Orden de importancia):
    1. ¿Incluyó la Advertencia de Seguridad obligatoria sobre el dolor?
    2. ¿Respetó estrictamente la restricción de NO USAR PESAS?
    3. ¿Evitó ejercicios que flexionen mucho la rodilla (ej. sentadillas profundas o saltos)?
    4. ¿Mantuvo una estructura clara y profesional en la respuesta? Manteniendo la división en Parámetros, Rutina y Justificación.
    
    OPCIÓN 1:
    {op1}
    ---
    OPCIÓN 2:
    {op2}
    ---
    OPCIÓN 3:
    {op3}
    
    Analiza brevemente las 3 opciones e indica ÚNICAMENTE el NÚMERO de la mejor opción (ejemplo: "OPCIÓN 2"). No transcribas la rutina.
    """
    
    # Pausa antes del Juez para asegurar que el bucket de tokens esté vacío
    time.sleep(10)
    
    res_juez = llm_judge.invoke(judge_template.format(question=pregunta, op1=respuestas[0][:], op2=respuestas[1][:], op3=respuestas[2][:]))
    print(f"\nVeredicto: {res_juez.content}")
    
    # Por ahora, para no complicar el código, devolvemos la primera para la demo
    return respuestas[0]

# --- EJECUCIÓN DEL BENCHMARK ---
if __name__ == "__main__":
    print("\n" + "*"*50)
    print("INICIANDO BENCHMARK DE PROMPT ENGINEERING")
    print("*"*50)
    
    print("\n1️⃣ TEST ZERO-SHOT:")
    print(clean_qwen_output(chain_zs.invoke(question)))
    
    print("\n2️⃣ TEST FEW-SHOT:")
    print(clean_qwen_output(chain_fs.invoke(question)))
    
    print("\n3️⃣ TEST CHAIN-OF-THOUGHT (CoT):")
    # En CoT NO limpiamos el output para poder ver el razonamiento del modelo en la consola
    print(chain_cot.invoke(question)) 
    
    print("\n4️⃣ TEST SELF-CONSISTENCY:")
    print(run_self_consistency(question))