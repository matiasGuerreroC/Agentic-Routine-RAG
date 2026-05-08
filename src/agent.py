import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from time import sleep

import torch
import logging
import random
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# Reducir ruido de logs de transformers en backend
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

CHROMA_PATH = "chromadb_storage"
DEFAULT_LLM_MODEL = "qwen/qwen3-32b"
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Control para limitar tamaño de contexto enviado al LLM (evita rate limits por tokens)
MAX_CONTEXT_DOCS = 6  # número máximo de documentos a incluir en el contexto
DOC_SNIPPET_CHARS = 1200  # caracteres por documento para el snippet


def clean_qwen_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"Fuente: {doc.metadata.get('source')}"
            f" | Idioma: {doc.metadata.get('language', 'unknown')}\n"
            f"Contenido: {doc.page_content}"
            for doc in docs
        ]
    )


def get_embeddings(device: Optional[str] = None) -> HuggingFaceEmbeddings:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": selected_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(
    chroma_path: str = CHROMA_PATH,
    embedding_function: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    embeddings = embedding_function or get_embeddings()
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


def get_retriever(
    chroma_path: str = CHROMA_PATH,
    embedding_function: Optional[HuggingFaceEmbeddings] = None,
    k: int = 4,
):
    db = get_vector_store(chroma_path=chroma_path, embedding_function=embedding_function)
    return db.as_retriever(search_kwargs={"k": k})


def create_llm(
    temperature: float,
    model_name: str = DEFAULT_LLM_MODEL,
    groq_api_key: Optional[str] = None,
) -> ChatGroq:
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
    )


def wait_for_continue(prompt: str) -> None:
    print(prompt)
    try:
        import msvcrt

        msvcrt.getch()
    except ImportError:
        input()


def translate_question_to_english(
    question_spanish: str,
    translator_llm: Optional[ChatGroq] = None,
) -> str:
    llm = translator_llm or create_llm(temperature=0.0)
    translate_prompt = ChatPromptTemplate.from_template(
        """
        Traduce del español al inglés de forma literal y clara para búsqueda semántica.
        Devuelve solo la traducción final en inglés, sin explicaciones.

        Texto en español:
        {question_spanish}
        """
    )
    chain = translate_prompt | llm | StrOutputParser()
    return clean_qwen_output(chain.invoke({"question_spanish": question_spanish}))


GENERATOR_TEMPLATE = """
Actúa como un entrenador personal experto basado estrictamente en evidencia científica deportiva.
Debes diseñar una rutina utilizando EXCLUSIVAMENTE el contexto proporcionado y el equipamiento disponible.

La pregunta original del usuario está en español y también tienes una traducción al inglés usada para recuperar evidencia.
Usa ambos campos para evitar ambigüedades.

REGLAS DE RAZONAMIENTO (Chain-of-Thought):
Antes de dar la rutina, haz un análisis explícito:
1. ¿Qué equipo tiene disponible el usuario?
2. ¿Tiene alguna lesión o dolor mencionado? (Si hay dolor, evita ejercicios que afecten esa zona).
3. ¿Qué volumen recomienda la evidencia recuperada?

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

PREGUNTA DEL USUARIO (ESPAÑOL):
{question_es}

TRADUCCIÓN USADA EN RETRIEVAL (INGLÉS):
{question_en}

TU RESPUESTA (en español, incluye ANÁLISIS seguido de las 3 secciones):
"""

JUDGE_TEMPLATE = """
Eres un supervisor clínico-deportivo. Revisa estas 3 opciones de rutina generadas para la siguiente petición:
PREGUNTA DEL USUARIO: "{question_es}"

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

# Template para evaluación de la Tríada de RAG
TRIAD_EVALUATION_TEMPLATE = """
Actúa como un Auditor Crítico de Sistemas RAG. Tu tarea es evaluar la calidad de una respuesta basada en la Tríada de RAG.

CRITERIOS DE EVALUACIÓN (Escala 1 a 5):
1. RELEVANCIA DEL CONTEXTO: ¿Los documentos recuperados contienen la información necesaria para responder?
2. FIDELIDAD (Faithfulness): ¿La respuesta se basa estrictamente en el contexto (sin inventar hechos)?
3. RELEVANCIA DE LA RESPUESTA: ¿La respuesta soluciona directamente la duda del usuario?

DATOS A EVALUAR:
- PREGUNTA DEL USUARIO: {question_es}
- CONTEXTO RECUPERADO: {context}
- RESPUESTA DEL SISTEMA: {answer}

Responde en formato JSON con esta estructura:
{{
  "relevancia_contexto": {número 1-5},
  "justificacion_contexto": "texto breve",
  "fidelidad": {número 1-5},
  "justificacion_fidelidad": "texto breve",
  "relevancia_respuesta": {número 1-5},
  "justificacion_respuesta": "texto breve",
  "score_general": {número 1-5},
  "recomendaciones": "observaciones clave"
}}
"""

# Template para Multi-Query Retrieval
MQR_TEMPLATE = """
Eres un experto en ciencias del deporte. Tu tarea es generar 3 versiones diferentes de la pregunta dada
en inglés técnico para recuperar documentos relevantes de una base de datos científica.

Enfatiza términos técnicos como: 'hypertrophy', 'mechanical tension', 'range of motion', 
'progressive overload', 'muscular endurance', 'strength training'.

Pregunta original en español: {question}

Genera EXACTAMENTE 3 variantes en inglés técnico, una por línea, sin numeración ni explicación:
"""


@dataclass
class RoutineRAGAgent:
    chroma_path: str = CHROMA_PATH
    k: int = 4
    llm_model_name: str = DEFAULT_LLM_MODEL

    def __post_init__(self) -> None:
        self._embeddings = get_embeddings()
        
        # Configurar retriever base
        db = get_vector_store(
            chroma_path=self.chroma_path,
            embedding_function=self._embeddings
        )
        retriever_base = db.as_retriever(search_kwargs={"k": self.k})
        
        # Configurar LLMs
        self._llm_generator = create_llm(
            temperature=0.4,
            model_name=self.llm_model_name,
        )
        self._llm_judge = create_llm(
            temperature=0.1,
            model_name=self.llm_model_name,
        )
        self._llm_translator = create_llm(
            temperature=0.0,
            model_name=self.llm_model_name,
        )
        # LLM auditor para evaluación de Tríada (Llama 3.3 70B es más riguroso)
        self._llm_auditor = create_llm(
            temperature=0.0,
            model_name="llama-3.3-70b-versatile",  # Auditor más potente
        )
        
        # Integrar Multi-Query Retrieval (MQR)
        mqr_prompt = ChatPromptTemplate.from_template(MQR_TEMPLATE)
        try:
            self._retriever = MultiQueryRetriever.from_llm(
                retriever=retriever_base,
                llm=self._llm_translator,
                prompt=mqr_prompt,
            )
        except Exception as e:
            print(f"⚠️ Advertencia: MQR no disponible ({str(e)}). Usando retriever base.")
            self._retriever = retriever_base
        
        # Chains para generación y juicio
        self._generator_chain = (
            ChatPromptTemplate.from_template(GENERATOR_TEMPLATE)
            | self._llm_generator
            | StrOutputParser()
        )
        self._judge_chain = (
            ChatPromptTemplate.from_template(JUDGE_TEMPLATE)
            | self._llm_judge
            | StrOutputParser()
        )
        # Chain para evaluación de Tríada
        self._triad_chain = (
            ChatPromptTemplate.from_template(TRIAD_EVALUATION_TEMPLATE)
            | self._llm_auditor
            | StrOutputParser()
        )

    def translate_question(self, question_spanish: str) -> str:
        return translate_question_to_english(
            question_spanish=question_spanish,
            translator_llm=self._llm_translator,
        )

    def retrieve_context(self, question_english: str) -> str:
        """Recupera contexto usando Multi-Query Retrieval (MQR).
        
        MQR automáticamente genera 3 variantes de la pregunta en inglés técnico
        antes de hacer búsqueda semántica, mejorando la calidad del retrieval.
        """
        try:
            docs = self._retriever.invoke(question_english)
            # Acortar el contexto: tomar sólo los primeros N documentos y recortar
            # cada documento a un snippet de longitud controlada. Esto reduce el
            # tamaño del prompt y ayuda a evitar límites de tokens / rate limits.
            try:
                top_docs = docs[:MAX_CONTEXT_DOCS]
            except Exception:
                top_docs = docs

            # Crear copias con snippets recortados
            shortened_docs = []
            for d in top_docs:
                # A veces `page_content` puede ser muy grande; recortamos para el prompt
                snippet = (d.page_content[:DOC_SNIPPET_CHARS] + "...") if len(d.page_content) > DOC_SNIPPET_CHARS else d.page_content
                # Crear un Document-like dict para formatear
                doc_copy = Document(page_content=snippet, metadata=d.metadata)
                shortened_docs.append(doc_copy)

            return format_docs(shortened_docs)
        except Exception as e:
            print(f"⚠️ Error en retrieve_context: {str(e)}")
            return "[Error: No se pudo recuperar contexto]"

    def generate_candidates(
        self,
        question_spanish: str,
        question_english: str,
        context: str,
        samples: int = 3,
    ) -> List[str]:
        """Genera múltiples candidatos de rutina (Self-Consistency).
        
        Implementa reintentos automáticos para manejar Rate Limits de Groq.
        """
        responses: List[str] = []
        max_retries = 5

        for i in range(samples):
            print(f"   Generando opción {i + 1}/{samples}...")

            for attempt in range(max_retries):
                try:
                    raw_response = self._generator_chain.invoke(
                        {
                            "context": context,
                            "question_es": question_spanish,
                            "question_en": question_english,
                        }
                    )
                    responses.append(clean_qwen_output(raw_response))
                    break  # Salir del loop de reintentos

                except Exception as e:
                    error_msg = str(e)
                    # Detectar Rate Limit o errores de red y aplicar backoff con jitter
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        base = 10
                        wait_time = base * (2 ** attempt) + random.uniform(0, 3)
                        print(f"   ⏳ Rate Limit detectado. Esperando {int(wait_time)}s (attempt {attempt + 1})...")
                        sleep(wait_time)
                    else:
                        # Backoff más corto para otros errores
                        wait_time = 2 * (attempt + 1) + random.uniform(0, 1)
                        print(f"   ⚠️ Error generando opción {i + 1} (attempt {attempt + 1}): {error_msg}. Esperando {int(wait_time)}s antes de reintentar.")
                        sleep(wait_time)

                    # Si es el último intento y no se pudo generar, añadir marcador de error
                    if attempt == max_retries - 1:
                        print(f"   ⚠️ Último intento fallido para opción {i + 1}: {error_msg}")
                        responses.append("[Error al generar rutina]")

        # Si no se generaron suficientes candidatos, rellenar con duplicados del primero
        if len(responses) < samples:
            if responses:
                last = responses[-1]
                while len(responses) < samples:
                    responses.append(last + "\n\n[Fallback duplicado por fallo de generación]")
            else:
                while len(responses) < samples:
                    responses.append("[Error al generar rutina]")

        return responses

    def judge_candidates(self, question_spanish: str, candidates: List[str]) -> str:
        """Evalúa 3 candidatos y retorna el mejor usando LLM Juez."""
        # Aceptar listas con menos de 3 candidatos rellenando con duplicados
        if len(candidates) < 3:
            if candidates:
                while len(candidates) < 3:
                    candidates.append(candidates[-1])
            else:
                # No hay candidatos, retornar marcador de error
                return "[Error: no se generaron candidatos]"

        try:
            best_raw = self._judge_chain.invoke(
                {
                    "question_es": question_spanish,
                    "op1": candidates[0],
                    "op2": candidates[1],
                    "op3": candidates[2],
                }
            )
            return clean_qwen_output(best_raw)
        except Exception as e:
            # Fallback: retornar el primero si hay error de Rate Limit
            print(f"⚠️ Error en judge_candidates: {str(e)}. Usando opción 1.")
            return candidates[0]

    def evaluate_rag_triad(
        self,
        question_spanish: str,
        answer: str,
        docs_retrieved: List[Document],
    ) -> Dict[str, Any]:
        """
        Evalúa la respuesta según la Tríada de RAG:
        - Relevancia del Contexto
        - Fidelidad (Faithfulness)
        - Relevancia de la Respuesta
        
        Retorna un diccionario con puntuaciones y justificaciones.
        """
        # Formatear contexto
        context_formatted = format_docs(docs_retrieved)
        
        try:
            evaluation_raw = self._triad_chain.invoke(
                {
                    "question_es": question_spanish,
                    "context": context_formatted,
                    "answer": answer,
                }
            )
            evaluation_clean = clean_qwen_output(evaluation_raw)
            
            # Intentar parsear como JSON
            import json
            try:
                evaluation_dict = json.loads(evaluation_clean)
            except json.JSONDecodeError:
                # Fallback: retornar estructura genérica
                evaluation_dict = {
                    "relevancia_contexto": 3,
                    "justificacion_contexto": "No se pudo evaluar",
                    "fidelidad": 3,
                    "justificacion_fidelidad": "No se pudo evaluar",
                    "relevancia_respuesta": 3,
                    "justificacion_respuesta": "No se pudo evaluar",
                    "score_general": 3,
                    "recomendaciones": "Evaluación automática falló",
                }
            
            return evaluation_dict
        except Exception as e:
            print(f"⚠️ Error en evaluate_rag_triad: {str(e)}")
            return {
                "error": str(e),
                "score_general": 0,
                "recomendaciones": "Evaluación no disponible",
            }

    def run_pipeline(self, question_spanish: str, samples: int = 3, include_rag_triad: bool = True):
        """
        Pipeline completo con Multi-Query Retrieval y evaluación opcional de Tríada RAG.
        
        Args:
            question_spanish: Pregunta del usuario en español
            samples: Número de candidatos a generar (default: 3)
            include_rag_triad: Si True, incluir evaluación de la Tríada de RAG
            
        Returns:
            Dict con question_spanish, question_english, context, candidates, final_answer,
            y opcionalmente rag_triad_evaluation
        """
        # Paso 0: Traducir
        question_english = self.translate_question(question_spanish)
        
        # Paso 1: Retrieval con MQR (genera variantes automáticamente)
        context = self.retrieve_context(question_english)
        
        # Extraer docs para evaluación (requiere acceso a docs_list si es MQR)
        docs_list = []
        try:
            if isinstance(self._retriever, MultiQueryRetriever):
                # Para MQR, hacemos un invoke directo para obtener docs
                docs_list = self._retriever.invoke(question_english)
            else:
                # Para retriever base
                docs_list = self._retriever.invoke(question_english)
        except:
            pass  # Si falla, continuamos sin docs_list para evaluación
        
        # Paso 2: Generación de candidatos
        candidates = self.generate_candidates(
            question_spanish=question_spanish,
            question_english=question_english,
            context=context,
            samples=samples,
        )
        
        # Paso 3: Juicio sobre candidatos
        final_answer = self.judge_candidates(question_spanish, candidates)
        
        # Paso 4: Evaluación de Tríada RAG (opcional, para reportes)
        result = {
            "question_spanish": question_spanish,
            "question_english": question_english,
            "context": context,
            "candidates": candidates,
            "final_answer": final_answer,
        }
        
        if include_rag_triad and docs_list:
            result["rag_triad_evaluation"] = self.evaluate_rag_triad(
                question_spanish=question_spanish,
                answer=final_answer,
                docs_retrieved=docs_list,
            )
        
        return result

    def generate_routine(self, question_spanish: str, samples: int = 3, include_rag_triad: bool = False) -> str:
        """Genera una rutina basada en la pregunta del usuario."""
        return self.run_pipeline(question_spanish, samples=samples, include_rag_triad=include_rag_triad)["final_answer"]

    def run_interactive_console(self, question_spanish: str, samples: int = 3) -> str:
        """
        Consola interactiva que muestra los pasos del pipeline RAG avanzado:
        0. Traducción (ES -> EN)
        1. Multi-Query Retrieval (generación de variantes técnicas)
        2. Recuperación de evidencia científica
        3. Generación de múltiples caminos (Self-Consistency)
        4. Evaluación y selección del mejor (LLM Juez)
        """
        print("[Paso 0] Traduciendo consulta ES -> EN para retrieval semántico...")
        question_english = self.translate_question(question_spanish)
        print(f"\nTraducción al inglés:\n{question_english}")
        wait_for_continue("\nPresiona una tecla para continuar al Paso 1...")

        print(
            "\n[Paso 1] Multi-Query Retrieval: Generando variantes técnicas en inglés..."
        )
        print("El sistema generará 3 formas distintas de preguntar lo mismo,")
        print("asegurando que encuentre la mejor evidencia disponible.\n")
        context = self.retrieve_context(question_english)
        print("Contexto recuperado (primeras 500 caracteres):")
        print(context[:500] + "...")
        wait_for_continue("\nPresiona una tecla para continuar al Paso 2...")

        print(
            "\n[Paso 2] Generando múltiples caminos de razonamiento (Self-Consistency)..."
        )
        candidates = self.generate_candidates(
            question_spanish=question_spanish,
            question_english=question_english,
            context=context,
            samples=samples,
        )
        for index, candidate in enumerate(candidates, start=1):
            print(f"\nOpción {index}:")
            print(candidate[:400] + "...\n")
        wait_for_continue("\nPresiona una tecla para continuar al Paso 3...")

        print(
            "\n[Paso 3] Evaluando la opción más segura y consistente (LLM Juez)..."
        )
        final_answer = self.judge_candidates(question_spanish, candidates)
        print("\nRespuesta final:")
        print(final_answer)
        return final_answer


def generar_rutina_robusta(
    pregunta_usuario: str,
    chroma_path: str = CHROMA_PATH,
    samples: int = 3,
) -> str:
    agent = RoutineRAGAgent(chroma_path=chroma_path)
    return agent.generate_routine(pregunta_usuario, samples=samples)


if __name__ == "__main__":
    pregunta = """
    Hola, soy principiante y quiero aumentar la masa muscular de mi tren superior y piernas
    entrenando en mi casa 3 veces por semana. Estrictamente NO tengo pesas ni mancuernas,
    solo puedo usar mi peso corporal (calistenia) y una silla.
    Además, me duele un poco la rodilla derecha al flectarla mucho.
    ¿Me puedes armar una rutina y explicarme cuántas series y descansos necesito según la ciencia?
    """

    print("INICIANDO PIPELINE DE AGENTES AVANZADO\n")
    agent = RoutineRAGAgent()
    rutina_final = agent.run_interactive_console(pregunta)

    print("\n" + "=" * 60)
    print("RUTINA FINAL GENERADA POR EL AGENTE:")
    print("=" * 60 + "\n")
    print(rutina_final)