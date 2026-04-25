import os
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = "chromadb_storage"
DEFAULT_LLM_MODEL = "qwen/qwen3-32b"
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


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


@dataclass
class RoutineRAGAgent:
    chroma_path: str = CHROMA_PATH
    k: int = 4
    llm_model_name: str = DEFAULT_LLM_MODEL

    def __post_init__(self) -> None:
        self._embeddings = get_embeddings()
        self._retriever = get_retriever(
            chroma_path=self.chroma_path,
            embedding_function=self._embeddings,
            k=self.k,
        )
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

    def translate_question(self, question_spanish: str) -> str:
        return translate_question_to_english(
            question_spanish=question_spanish,
            translator_llm=self._llm_translator,
        )

    def retrieve_context(self, question_english: str) -> str:
        docs = self._retriever.invoke(question_english)
        return format_docs(docs)

    def generate_candidates(
        self,
        question_spanish: str,
        question_english: str,
        context: str,
        samples: int = 3,
    ) -> List[str]:
        responses: List[str] = []
        for i in range(samples):
            print(f"   Generando opción {i + 1}...")
            raw_response = self._generator_chain.invoke(
                {
                    "context": context,
                    "question_es": question_spanish,
                    "question_en": question_english,
                }
            )
            responses.append(clean_qwen_output(raw_response))
        return responses

    def judge_candidates(self, question_spanish: str, candidates: List[str]) -> str:
        if len(candidates) < 3:
            raise ValueError("Se requieren al menos 3 candidatos para la evaluación.")

        best_raw = self._judge_chain.invoke(
            {
                "question_es": question_spanish,
                "op1": candidates[0],
                "op2": candidates[1],
                "op3": candidates[2],
            }
        )
        return clean_qwen_output(best_raw)

    def run_pipeline(self, question_spanish: str, samples: int = 3):
        question_english = self.translate_question(question_spanish)
        context = self.retrieve_context(question_english)
        candidates = self.generate_candidates(
            question_spanish=question_spanish,
            question_english=question_english,
            context=context,
            samples=samples,
        )
        final_answer = self.judge_candidates(question_spanish, candidates)
        return {
            "question_spanish": question_spanish,
            "question_english": question_english,
            "context": context,
            "candidates": candidates,
            "final_answer": final_answer,
        }

    def generate_routine(self, question_spanish: str, samples: int = 3) -> str:
        return self.run_pipeline(question_spanish, samples=samples)["final_answer"]

    def run_interactive_console(self, question_spanish: str, samples: int = 3) -> str:
        print("[Paso 0] Traduciendo consulta ES -> EN para retrieval semántico...")
        question_english = self.translate_question(question_spanish)
        print("\nTraducción al inglés:")
        print(question_english)
        wait_for_continue("\nPresiona una tecla para continuar al Paso 1...")

        print("\n[Paso 1] Recuperando evidencia desde la base vectorial...")
        context = self.retrieve_context(question_english)
        print("\nContexto recuperado:")
        print(context)
        wait_for_continue("\nPresiona una tecla para continuar al Paso 2...")

        print("\n[Paso 2] Generando múltiples caminos de razonamiento (Self-Consistency)...")
        candidates = self.generate_candidates(
            question_spanish=question_spanish,
            question_english=question_english,
            context=context,
            samples=samples,
        )
        for index, candidate in enumerate(candidates, start=1):
            print(f"\nOpción {index}:")
            print(candidate)
        wait_for_continue("\nPresiona una tecla para continuar al Paso 3...")

        print("\n[Paso 3] Evaluando la opción más segura y consistente (LLM Juez)...")
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