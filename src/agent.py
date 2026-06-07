from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

CHROMA_PATH = "chromadb_storage"
DEFAULT_LLM_MODEL = "qwen/qwen3-32b"
DEFAULT_SYNTHESIS_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

MAX_CONTEXT_DOCS = 6
DOC_SNIPPET_CHARS = 1200
MAX_TRIAD_CONTEXT_DOCS = 2
TRIAD_SNIPPET_CHARS = 600

WGER_API_BASE = "https://wger.de/api/v2/exerciseinfo/"
WGER_TIMEOUT_SECONDS = 20
WGER_RESULT_LIMIT = 8
WGER_LANGUAGE_SPANISH = 4

LLM_RETRY_ATTEMPTS = 4
LLM_RETRY_BASE_SLEEP = 2.0
TASK_PAUSE_SECONDS = 0.75
LLM_FINAL_CONTEXT_LIMIT = 360
LLM_FINAL_TRACE_LIMIT = 120

WGER_EQUIPMENT_ALIASES: Dict[str, int] = {
    "peso corporal": 7,
    "bodyweight": 7,
    "sin equipo": 7,
    "calistenia": 7,
    "silla": 8,
    "silla de casa": 8,
    "banco": 8,
    "bench": 8,
    "banco inclinado": 9,
    "inclined bench": 9,
    "mancuerna": 3,
    "mancuernas": 3,
    "dumbbell": 3,
    "colchoneta": 4,
    "esterilla": 4,
    "mat": 4,
    "fitball": 5,
    "pelota suiza": 5,
    "barra dominadas": 6,
    "barra": 1,
    "barbell": 1,
    "banda elastica": 11,
    "banda elástica": 11,
    "resistance band": 11,
    "kettlebell": 10,
    "pesa rusa": 10,
}

FALLBACK_WGER_EXERCISES: Dict[int, List[Dict[str, Any]]] = {
    7: [
        {"nombre": "Sentadilla al aire", "descripcion": "Baja con control manteniendo el tronco estable y vuelve a la posición inicial.", "categoria": "Piernas", "musculos": ["cuádriceps", "glúteos"], "equipamiento": [7], "language": 4},
        {"nombre": "Puente de glúteos", "descripcion": "Eleva la cadera apretando glúteos sin hiperextender la zona lumbar.", "categoria": "Cadera y glúteos", "musculos": ["glúteos", "isquiotibiales"], "equipamiento": [7], "language": 4},
        {"nombre": "Zancada estática", "descripcion": "Desciende de forma vertical con apoyo si hace falta y controla la rodilla delantera.", "categoria": "Piernas", "musculos": ["cuádriceps", "glúteos"], "equipamiento": [7], "language": 4},
        {"nombre": "Plancha frontal", "descripcion": "Mantén una línea recta desde la cabeza hasta los talones durante el tiempo indicado.", "categoria": "Core", "musculos": ["abdominales", "glúteos"], "equipamiento": [7], "language": 4},
    ],
    8: [
        {"nombre": "Step-up a silla", "descripcion": "Sube de manera controlada a una superficie estable y regresa sin impulso.", "categoria": "Piernas", "musculos": ["cuádriceps", "glúteos"], "equipamiento": [8], "language": 4},
        {"nombre": "Flexión inclinada con silla", "descripcion": "Apoya las manos en la silla para reducir carga y realiza una flexión controlada.", "categoria": "Tren superior", "musculos": ["pectorales", "tríceps"], "equipamiento": [8], "language": 4},
        {"nombre": "Hip thrust en silla", "descripcion": "Apoya la espalda superior en la silla y eleva la cadera con contracción de glúteos.", "categoria": "Cadera y glúteos", "musculos": ["glúteos", "isquiotibiales"], "equipamiento": [8], "language": 4},
    ],
    11: [
        {"nombre": "Remo con banda", "descripcion": "Tira de la banda hacia el torso con escápulas retraídas y columna neutra.", "categoria": "Espalda", "musculos": ["dorsales", "romboides", "bíceps"], "equipamiento": [11], "language": 4},
        {"nombre": "Press de pecho con banda", "descripcion": "Empuja la banda al frente sin elevar hombros y controla el retorno.", "categoria": "Pecho", "musculos": ["pectorales", "tríceps"], "equipamiento": [11], "language": 4},
        {"nombre": "Face pull con banda", "descripcion": "Lleva la banda hacia la cara manteniendo codos altos y hombros estables.", "categoria": "Hombros", "musculos": ["deltoides posteriores", "trapecio"], "equipamiento": [11], "language": 4},
    ],
    3: [
        {"nombre": "Sentadilla goblet", "descripcion": "Sujeta la mancuerna al pecho y baja con control sin perder postura.", "categoria": "Piernas", "musculos": ["cuádriceps", "glúteos"], "equipamiento": [3], "language": 4},
        {"nombre": "Press de hombro con mancuerna", "descripcion": "Empuja sobre la cabeza con recorrido controlado y costillas neutras.", "categoria": "Hombros", "musculos": ["deltoides", "tríceps"], "equipamiento": [3], "language": 4},
    ],
    1: [
        {"nombre": "Peso muerto rumano con barra", "descripcion": "Desliza la barra cerca del cuerpo manteniendo la cadera atrás y espalda neutra.", "categoria": "Cadena posterior", "musculos": ["isquiotibiales", "glúteos"], "equipamiento": [1], "language": 4},
    ],
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", normalized.lower()).strip()


def detect_rate_limit_error(error: Exception | str) -> bool:
    message = str(error).lower()
    return any(token in message for token in ["429", "rate_limit", "rate limit", "tpm", "too many requests"])


def clean_qwen_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def trim_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[recortado]"


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"Fuente: {doc.metadata.get('source')} | Idioma: {doc.metadata.get('language', 'unknown')}\nContenido: {doc.page_content}"
            for doc in docs
        ]
    )


def extract_json_block(text: str) -> Optional[Any]:
    cleaned = clean_qwen_output(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for candidate in re.findall(r"\[[\s\S]*\]", cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in re.findall(r"\{[\s\S]*\}", cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def extract_json_array_from_text(text: str) -> List[Dict[str, Any]]:
    parsed = extract_json_block(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    match = re.search(r"\[[\s\S]*\]", clean_qwen_output(text))
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    return []


def compress_context_for_triad(docs: List[Document]) -> str:
    top_docs = docs[:MAX_TRIAD_CONTEXT_DOCS]
    shortened_docs = []
    for doc in top_docs:
        snippet = doc.page_content[:TRIAD_SNIPPET_CHARS]
        if len(doc.page_content) > TRIAD_SNIPPET_CHARS:
            snippet += "..."
        shortened_docs.append(Document(page_content=snippet, metadata=doc.metadata))
    return format_docs(shortened_docs)


def get_embeddings(device: Optional[str] = None) -> HuggingFaceEmbeddings:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": selected_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(chroma_path: str = CHROMA_PATH, embedding_function: Optional[HuggingFaceEmbeddings] = None) -> Chroma:
    embeddings = embedding_function or get_embeddings()
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


def get_retriever(chroma_path: str = CHROMA_PATH, embedding_function: Optional[HuggingFaceEmbeddings] = None, k: int = 4):
    db = get_vector_store(chroma_path=chroma_path, embedding_function=embedding_function)
    return db.as_retriever(search_kwargs={"k": k})


def create_llm(temperature: float, model_name: str = DEFAULT_LLM_MODEL, groq_api_key: Optional[str] = None) -> ChatGroq:
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


def safe_runnable_invoke(runnable: Any, payload: Any, retries: int = LLM_RETRY_ATTEMPTS) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            last_error = exc
            if detect_rate_limit_error(exc):
                wait_time = LLM_RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0.5, 2.0)
            else:
                wait_time = 1.5 * (attempt + 1) + random.uniform(0.2, 1.0)
            print(f"⚠️ Error LLM (intento {attempt + 1}/{retries}): {exc}. Pausando {wait_time:.1f}s.")
            time.sleep(wait_time)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Error desconocido al invocar el LLM.")


def detect_pain_signal(question_spanish: str) -> Dict[str, Any]:
    normalized = normalize_text(question_spanish)
    pain_words = ["dolor", "molestia", "molestias", "lesion", "lesión", "inflamacion", "inflamación", "sensible"]
    zone_keywords = ["rodilla", "espalda baja", "zona lumbar", "lumbar", "hombro", "codo", "muñeca", "cuello", "cadera", "tobillo", "mano"]
    has_pain = any(word in normalized for word in pain_words)
    zones = [zone for zone in zone_keywords if zone in normalized]
    return {"has_pain": has_pain, "zones": zones}


def extract_days_per_week(question_spanish: str) -> Optional[int]:
    normalized = normalize_text(question_spanish)
    patterns = [
        r"(\d+)\s*(?:dias?|d[ií]as?)\s+por\s+semana",
        r"(\d+)\s*(?:veces?|sesiones?)\s+por\s+semana",
        r"(\d+)\s*(?:veces?|sesiones?)\s+semana",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def build_safety_warning(question_spanish: str) -> str:
    pain = detect_pain_signal(question_spanish)
    if not pain["has_pain"]:
        return ""
    zone_text = ", ".join(sorted(set(pain["zones"]))) if pain["zones"] else "la zona afectada"
    return (
        "**⚠️ ADVERTENCIA DE SEGURIDAD:** Si presentas dolor o molestia persistente, detén el ejercicio y consulta a un profesional de la salud. "
        f"Esta rutina evitará ejercicios que carguen directamente {zone_text}.\n\n"
    )


def clean_html_to_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def select_spanish_translation(item: Dict[str, Any]) -> Dict[str, Any]:
    translations = item.get("translations", []) or []
    spanish = next((translation for translation in translations if translation.get("language") == WGER_LANGUAGE_SPANISH), None)
    fallback = translations[0] if translations else {}
    chosen = spanish or fallback
    return {
        "id": item.get("id"),
        "nombre": chosen.get("name") or item.get("id"),
        "descripcion": clean_html_to_text(chosen.get("description") or ""),
        "categoria": item.get("category", {}).get("name") if isinstance(item.get("category"), dict) else item.get("category"),
        "musculos": [muscle.get("name") if isinstance(muscle, dict) else muscle for muscle in item.get("muscles", [])],
        "musculos_secundarios": [muscle.get("name") if isinstance(muscle, dict) else muscle for muscle in item.get("muscles_secondary", [])],
        "equipamiento": [equip.get("name") if isinstance(equip, dict) else equip for equip in item.get("equipment", [])],
        "language": chosen.get("language", WGER_LANGUAGE_SPANISH),
    }


def summarize_docs_for_state(docs: List[Document]) -> str:
    clipped_docs = []
    for doc in docs[:MAX_CONTEXT_DOCS]:
        snippet = doc.page_content[:DOC_SNIPPET_CHARS]
        if len(doc.page_content) > DOC_SNIPPET_CHARS:
            snippet += "..."
        clipped_docs.append(Document(page_content=snippet, metadata=doc.metadata))
    return format_docs(clipped_docs)


PLANNER_PROMPT = """
Eres un arquitecto planificador de rutinas de entrenamiento.

Debes analizar la consulta del usuario en español, traducirla internamente al inglés si lo necesitas para razonar mejor, e identificar:
1. El equipamiento disponible.
2. Las molestias o dolores físicos mencionados.
3. La evidencia que hace falta consultar.

Reglas de salida:
- Devuelve ÚNICAMENTE un JSON Array válido.
- Cada elemento debe tener exactamente estas claves: "paso", "tarea", "tool".
- "tool" debe ser una de estas opciones: "consultar_rag", "consultar_wger_api", "ninguna".
- El array debe estar ordenado de menor a mayor según "paso".
- Si hay dolor o molestia, incluye una tarea explícita de seguridad antes de la síntesis final.
- No agregues texto fuera del JSON.

Ejemplo de formato:
[
    {{"paso": 1, "tarea": "buscar evidencia sobre volumen para principiante en casa", "tool": "consultar_rag"}},
    {{"paso": 2, "tarea": "obtener ejercicios compatibles con el equipamiento disponible", "tool": "consultar_wger_api"}},
    {{"paso": 3, "tarea": "sintetizar una rutina segura y personalizada", "tool": "ninguna"}}
]

Consulta del usuario:
{question_es}
"""


EXECUTOR_SYSTEM_PROMPT = """
Eres el ejecutor de una arquitectura planificadora.

Tu tarea es mirar la subtarea actual, decidir si necesitas usar una herramienta y luego integrar su resultado en el estado.

Herramientas disponibles:
- consultar_rag: busca evidencia científica en ChromaDB usando una consulta en inglés.
- consultar_wger_api: obtiene ejercicios desde Wger filtrados por equipamiento en español.

Reglas:
- Si la subtarea requiere evidencia, usa una herramienta.
- Si la subtarea es de integración o seguridad, puedes responder sin herramienta.
- Cuando respondas después de una herramienta, sé breve y devuelve un resumen operativo en español.
- No inventes datos que no estén en el estado ni en la respuesta de la herramienta.
"""


FINAL_SYNTHESIS_PROMPT = """
Genera una rutina de entrenamiento en español y solo en Markdown.

Reglas mínimas:
- Si hay advertencia de seguridad, inclúyela al inicio.
- Usa exactamente 3 secciones: 1. Parámetros de Entrenamiento, 2. Rutina Propuesta, 3. Justificación Científica.
- Si hay días detectados, organiza por días; si no, usa lista secuencial.
- Cada ejercicio debe incluir: *Músculos objetivos*, *Cómo hacerlo* y *Modificación*.
- No inventes ejercicios ni volumen fuera del contexto.

Consulta breve:
{question_es}

Seguridad:
{safety_warning}

Días/semana:
{days_per_week}

Plan resumido:
{plan_summary}

Traces resumidas:
{task_results}

Evidencia resumida:
{scientific_context}

Wger resumido:
{wger_context}
"""


JUDGE_TEMPLATE = """
Eres un supervisor clínico-deportivo experto. Tu tarea es elegir la MEJOR rutina de las 3 opciones.

Criterios de evaluación (en orden de importancia):
1. SEGURIDAD: ¿Respeta lesiones/dolores mencionados? ¿Usa solo el equipo disponible?
2. CALIDAD: ¿Es científicamente rigurosa? ¿Tiene parámetros (series, reps, descanso)?
3. CLARIDAD: ¿Formato Markdown limpio? ¿Es entendible?

PREGUNTA DEL USUARIO: "{question_es}"

OPCIÓN 1:
{op1}
---
OPCIÓN 2:
{op2}
---
OPCIÓN 3:
{op3}

Análisis:
1. ¿Cuál respeta mejor la seguridad?
2. ¿Cuál tiene mejor calidad científica?
3. ¿Cuál es más clara?

Devuelve ÚNICAMENTE el número (1, 2 o 3) de la mejor opción seguido del texto completo de esa rutina, sin comentarios adicionales.
Formato: [NÚMERO]\n[RUTINA COMPLETA]
"""


TRIAD_EVALUATION_TEMPLATE = """Eres un Auditor Crítico de Sistemas RAG. Evalúa esta respuesta usando la Tríada de RAG.

CRITERIOS (escala 1-5):
1. Relevancia Contexto: ¿Los docs tienen la info necesaria?
2. Fidelidad: ¿La respuesta cita el contexto sin inventar?
3. Relevancia Respuesta: ¿Contesta directamente la pregunta?

PREGUNTA: {question_es}
CONTEXTO: {context}
RESPUESTA: {answer}

Responde SOLO con JSON válido, sin texto adicional:
{{
  "relevancia_contexto": (1-5),
  "justificacion_contexto": "texto breve",
  "fidelidad": (1-5),
  "justificacion_fidelidad": "texto breve",
  "relevancia_respuesta": (1-5),
  "justificacion_respuesta": "texto breve",
  "score_general": (1-5),
  "recomendaciones": "observaciones"
}}
"""


MQR_TEMPLATE = """
Eres un experto en ciencias del deporte. Tu tarea es generar 3 versiones diferentes de la pregunta dada
en inglés técnico para recuperar documentos relevantes de una base de datos científica.

Enfatiza términos técnicos como: 'hypertrophy', 'mechanical tension', 'range of motion',
'progressive overload', 'muscular endurance', 'strength training'.

Pregunta original en español: {question}

Genera EXACTAMENTE 3 variantes en inglés técnico, una por línea, sin numeración ni explicación:
"""


def build_fallback_wger_payload(equipment_id: int, equipment_term: str) -> Dict[str, Any]:
    exercises = FALLBACK_WGER_EXERCISES.get(equipment_id) or FALLBACK_WGER_EXERCISES.get(7, [])
    return {
        "source": "fallback",
        "language": WGER_LANGUAGE_SPANISH,
        "equipment_query": equipment_term,
        "equipment_id": equipment_id,
        "count": len(exercises),
        "results": exercises,
    }


def safe_runnable_invoke(runnable: Any, payload: Any, retries: int = LLM_RETRY_ATTEMPTS) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            last_error = exc
            if detect_rate_limit_error(exc):
                wait_time = LLM_RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0.5, 2.0)
            else:
                wait_time = 1.5 * (attempt + 1) + random.uniform(0.2, 1.0)
            print(f"⚠️ Error LLM (intento {attempt + 1}/{retries}): {exc}. Pausando {wait_time:.1f}s.")
            time.sleep(wait_time)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Error desconocido al invocar el LLM.")


@dataclass
class RoutineRAGAgent:
    chroma_path: str = CHROMA_PATH
    k: int = 4
    llm_model_name: str = DEFAULT_LLM_MODEL
    llm_synthesis_model_name: str = DEFAULT_SYNTHESIS_MODEL

    def __post_init__(self) -> None:
        self._embeddings = get_embeddings()
        self._vector_store = get_vector_store(self.chroma_path, self._embeddings)
        self._retriever_base = self._vector_store.as_retriever(search_kwargs={"k": self.k})

        self._llm_planner = create_llm(temperature=0.0, model_name=self.llm_model_name)
        self._llm_executor = create_llm(temperature=0.2, model_name=self.llm_model_name)
        self._llm_synthesizer = create_llm(temperature=0.3, model_name=self.llm_synthesis_model_name)
        self._llm_generator = create_llm(temperature=0.4, model_name=self.llm_model_name)
        self._llm_judge = create_llm(temperature=0.1, model_name=self.llm_synthesis_model_name)
        self._llm_translator = create_llm(temperature=0.0, model_name=self.llm_model_name)

        try:
            self._llm_auditor = create_llm(temperature=0.0, model_name="llama-3.3-70b-versatile")
        except Exception as exc:
            print(f"⚠️ No se pudo crear LLM auditor especializado ({exc}), usando LLM juez como fallback.")
            self._llm_auditor = self._llm_judge

        mqr_prompt = ChatPromptTemplate.from_template(MQR_TEMPLATE)
        try:
            self._retriever = MultiQueryRetriever.from_llm(retriever=self._retriever_base, llm=self._llm_translator, prompt=mqr_prompt)
        except Exception as exc:
            print(f"⚠️ Advertencia: MQR no disponible ({exc}). Usando retriever base.")
            self._retriever = self._retriever_base

        self._generator_chain = ChatPromptTemplate.from_template(GENERATOR_TEMPLATE) | self._llm_generator | StrOutputParser()
        self._judge_chain = ChatPromptTemplate.from_template(JUDGE_TEMPLATE) | self._llm_judge | StrOutputParser()
        try:
            self._triad_chain = ChatPromptTemplate.from_template(TRIAD_EVALUATION_TEMPLATE) | self._llm_auditor | StrOutputParser()
        except Exception as exc:
            print(f"⚠️ No se pudo crear triad_chain ({exc}). La evaluación LLM de la Tríada se deshabilita.")
            self._triad_chain = None

        self.tools = self._build_tools()
        self._tool_by_name = {item.name: item for item in self.tools}

        try:
            self._executor_with_tools = self._llm_executor.bind_tools(self.tools)
        except Exception as exc:
            print(f"⚠️ No se pudo activar bind_tools en el ejecutor ({exc}). Se usará fallback manual.")
            self._executor_with_tools = self._llm_executor

    def _build_tools(self) -> List[Any]:
        agent = self

        @tool("consultar_rag")
        def consultar_rag(termino_busqueda_en: str) -> str:
            """Consulta ChromaDB usando un término en inglés y devuelve el contexto recuperado."""
            return json.dumps(agent._consultar_rag_impl(termino_busqueda_en), ensure_ascii=False)

        @tool("consultar_wger_api")
        def consultar_wger_api(equipamiento_es: str) -> str:
            """Consulta la API de Wger usando equipamiento en español y devuelve ejercicios estructurados."""
            return json.dumps(agent._consultar_wger_api_impl(equipamiento_es), ensure_ascii=False)

        return [consultar_rag, consultar_wger_api]

    def _consultar_rag_impl(self, termino_busqueda_en: str) -> Dict[str, Any]:
        docs = self._retriever_base.invoke(termino_busqueda_en)
        context = summarize_docs_for_state(docs)
        return {
            "query_english": termino_busqueda_en,
            "doc_count": len(docs),
            "context": context,
            "documents": [
                {"source": doc.metadata.get("source"), "language": doc.metadata.get("language", "unknown"), "content": doc.page_content[:DOC_SNIPPET_CHARS]}
                for doc in docs[:MAX_CONTEXT_DOCS]
            ],
        }

    def _resolve_equipment_ids(self, equipamiento_es: str) -> List[Dict[str, Any]]:
        normalized = normalize_text(equipamiento_es)
        if not normalized:
            return [{"term": "peso corporal", "equipment_id": 7}]

        detected: List[Dict[str, Any]] = []
        for alias, equipment_id in WGER_EQUIPMENT_ALIASES.items():
            if alias in normalized:
                detected.append({"term": alias, "equipment_id": equipment_id})

        if detected:
            unique: Dict[int, Dict[str, Any]] = {}
            for item in detected:
                unique[item["equipment_id"]] = item
            return list(unique.values())

        tokens = [token.strip() for token in re.split(r"[,;/|]| y | con ", normalized) if token.strip()]
        for token in tokens:
            for alias, equipment_id in WGER_EQUIPMENT_ALIASES.items():
                if token == alias or token in alias or alias in token:
                    detected.append({"term": token, "equipment_id": equipment_id})

        if detected:
            unique = {}
            for item in detected:
                unique[item["equipment_id"]] = item
            return list(unique.values())

        return [{"term": equipamiento_es.strip() or "peso corporal", "equipment_id": 7}]

    def _fetch_wger_exercises(self, equipment_id: int) -> List[Dict[str, Any]]:
        params = {"language": WGER_LANGUAGE_SPANISH, "equipment": equipment_id, "limit": WGER_RESULT_LIMIT}
        response = requests.get(WGER_API_BASE, params=params, timeout=WGER_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        exercises: List[Dict[str, Any]] = []
        for item in payload.get("results", []):
            if isinstance(item, dict):
                exercises.append(select_spanish_translation(item))
        return exercises

    def _consultar_wger_api_impl(self, equipamiento_es: str) -> Dict[str, Any]:
        equipment_matches = self._resolve_equipment_ids(equipamiento_es)
        aggregated_results: List[Dict[str, Any]] = []
        source = "api"
        errors: List[str] = []

        for match in equipment_matches:
            equipment_id = int(match["equipment_id"])
            try:
                exercises = self._fetch_wger_exercises(equipment_id)
                if exercises:
                    aggregated_results.extend(exercises)
            except Exception as exc:
                source = "fallback"
                errors.append(f"equipment_id={equipment_id}: {exc}")
                fallback_payload = build_fallback_wger_payload(equipment_id, match.get("term", equipamiento_es))
                aggregated_results.extend(fallback_payload.get("results", []))

        if not aggregated_results:
            source = "fallback"
            fallback_payload = build_fallback_wger_payload(7, equipamiento_es)
            aggregated_results = fallback_payload.get("results", [])

        deduplicated: Dict[str, Dict[str, Any]] = {}
        for item in aggregated_results:
            key = normalize_text(str(item.get("nombre", ""))) or str(item.get("id", random.randint(1000, 9999)))
            deduplicated[key] = item

        return {
            "source": source,
            "language": WGER_LANGUAGE_SPANISH,
            "equipment_query": equipamiento_es,
            "equipment_matches": equipment_matches,
            "count": len(deduplicated),
            "errors": errors,
            "results": list(deduplicated.values())[:WGER_RESULT_LIMIT],
        }

    def _plan_prompt(self, question_spanish: str) -> str:
        return PLANNER_PROMPT.format(question_es=question_spanish.strip())

    def _executor_messages(self, task: Dict[str, Any], state: Dict[str, Any]) -> List[Any]:
        state_summary = {
            "question_es": state.get("question_es"),
            "question_en": state.get("question_en"),
            "has_pain": state.get("safety", {}).get("has_pain", False),
            "pain_zones": state.get("safety", {}).get("zones", []),
            "equipment_text": state.get("equipment_text"),
            "available_tools": [item.name for item in self.tools],
            "latest_rag_context": trim_text(state.get("scientific_context", ""), 1200),
            "latest_wger_context": trim_text(state.get("wger_context", ""), 1200),
        }
        user_prompt = (
            f"Subtarea actual: {json.dumps(task, ensure_ascii=False)}\n\n"
            f"Estado disponible: {json.dumps(state_summary, ensure_ascii=False)}\n\n"
            "Decide si necesitas usar una herramienta. Si la usas, responde con tool calling. Si no la usas, devuelve un resumen breve y operativo de la subtarea en español."
        )
        return [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    def _invoke_executor_step(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._executor_messages(task, state)
        ai_message = safe_runnable_invoke(self._executor_with_tools, messages)

        tool_results: List[Dict[str, Any]] = []
        trace_events: List[Dict[str, Any]] = []
        if isinstance(ai_message, AIMessage) and getattr(ai_message, "tool_calls", None):
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {}) or {}
                tool = self._tool_by_name.get(tool_name)
                if tool is None:
                    continue

                trace_events.append(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "arguments": tool_args,
                    }
                )

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"input": tool_args}

                tool_output = tool.invoke(tool_args)
                tool_results.append({"tool": tool_name, "args": tool_args, "output": tool_output})
                trace_events.append(
                    {
                        "type": "tool_message",
                        "tool": tool_name,
                        "content": tool_output,
                    }
                )

                follow_up_messages = messages + [ai_message, ToolMessage(content=str(tool_output), tool_call_id=tool_call.get("id", "tool-call"))]
                follow_up = safe_runnable_invoke(self._executor_with_tools, follow_up_messages)
                follow_up_text = follow_up.content if isinstance(follow_up, AIMessage) else str(follow_up)

                return {"task": task, "assistant_summary": clean_qwen_output(follow_up_text), "tool_results": tool_results, "trace_events": trace_events}

        fallback_text = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
        trace_events.append(
            {
                "type": "assistant_message",
                "content": clean_qwen_output(fallback_text),
            }
        )
        return {"task": task, "assistant_summary": clean_qwen_output(fallback_text), "tool_results": tool_results, "trace_events": trace_events}

    def _summarize_plan(self, plan: List[Dict[str, Any]]) -> str:
        compact_plan = []
        for item in plan[:3]:
            compact_plan.append(
                {
                    "paso": item.get("paso"),
                    "tool": item.get("tool"),
                    "tarea": trim_text(str(item.get("tarea", "")), 60),
                }
            )
        return json.dumps(compact_plan, ensure_ascii=False)

    def _summarize_execution_trace(self, execution_trace: List[Dict[str, Any]]) -> str:
        compact_steps: List[Dict[str, Any]] = []
        for step in execution_trace[:3]:
            task = step.get("task", {})
            tool_names = [item.get("tool") for item in step.get("tool_results", []) if item.get("tool")]
            compact_steps.append(
                {
                    "paso": task.get("paso"),
                    "tarea": task.get("tarea"),
                    "tool": task.get("tool"),
                    "herramientas_usadas": tool_names,
                    "resumen": trim_text(step.get("assistant_summary", ""), LLM_FINAL_TRACE_LIMIT),
                }
            )
        return json.dumps(compact_steps, ensure_ascii=False, indent=2)

    def _compact_context_excerpt(self, text: str, max_chars: int = 360) -> str:
        if not text:
            return "[Sin contexto]"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        excerpt = " | ".join(lines[:3]) if lines else text
        return trim_text(excerpt, max_chars)

    def _synthesize_final_routine(self, state: Dict[str, Any]) -> str:
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            question_es=trim_text(state.get("question_es", ""), 220),
            safety_warning=state.get("safety_warning", ""),
            days_per_week=state.get("days_per_week") or "No detectados",
            plan_summary=self._summarize_plan(state.get("plan", [])),
            task_results=self._summarize_execution_trace(state.get("execution_trace", [])),
            scientific_context=self._compact_context_excerpt(state.get("scientific_context", ""), LLM_FINAL_CONTEXT_LIMIT),
            wger_context=self._compact_context_excerpt(state.get("wger_context", ""), LLM_FINAL_CONTEXT_LIMIT),
        )
        final_raw = safe_runnable_invoke(self._llm_synthesizer, prompt)
        final_text = final_raw.content if isinstance(final_raw, AIMessage) else str(final_raw)
        final_text = clean_qwen_output(final_text)
        safety_warning = state.get("safety_warning", "")
        if safety_warning and not final_text.startswith("**⚠️ ADVERTENCIA DE SEGURIDAD:**"):
            final_text = safety_warning + final_text
        return final_text

    def translate_question(self, question_spanish: str) -> str:
        translation_prompt = ChatPromptTemplate.from_template(
            """
            Traduce del español al inglés de forma literal y clara para búsqueda semántica.
            Devuelve solo la traducción final en inglés, sin explicaciones.

            Texto en español:
            {question_spanish}
            """
        )
        chain = translation_prompt | self._llm_translator | StrOutputParser()
        translated = safe_runnable_invoke(chain, {"question_spanish": question_spanish})
        return clean_qwen_output(str(translated))

    def retrieve_context(self, question_english: str) -> str:
        try:
            raw = self._consultar_rag_impl(question_english)
            return raw["context"]
        except Exception as exc:
            print(f"⚠️ Error en retrieve_context: {exc}")
            return "[Error: No se pudo recuperar contexto]"

    def generate_candidates(self, question_spanish: str, question_english: str, context: str, samples: int = 3) -> List[str]:
        responses: List[str] = []
        max_retries = 5

        for i in range(samples):
            print(f"   Generando opción {i + 1}/{samples}...")
            for attempt in range(max_retries):
                try:
                    raw_response = safe_runnable_invoke(
                        self._generator_chain,
                        {"context": context, "question_es": question_spanish, "question_en": question_english},
                    )
                    responses.append(clean_qwen_output(str(raw_response)))
                    break
                except Exception as exc:
                    error_msg = str(exc)
                    if detect_rate_limit_error(error_msg):
                        wait_time = 10 * (2 ** attempt) + random.uniform(0, 3)
                        print(f"   ⏳ Rate Limit detectado. Esperando {int(wait_time)}s (attempt {attempt + 1})...")
                        time.sleep(wait_time)
                    else:
                        wait_time = 2 * (attempt + 1) + random.uniform(0, 1)
                        print(f"   ⚠️ Error generando opción {i + 1} (attempt {attempt + 1}): {error_msg}. Esperando {int(wait_time)}s antes de reintentar.")
                        time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print(f"   ⚠️ Último intento fallido para opción {i + 1}: {error_msg}")
                        responses.append("[Error al generar rutina]")

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
        if len(candidates) < 3:
            if candidates:
                while len(candidates) < 3:
                    candidates.append(candidates[-1])
            else:
                return "[Error: no se generaron candidatos]"

        try:
            best_raw = safe_runnable_invoke(self._judge_chain, {"question_es": question_spanish, "op1": candidates[0], "op2": candidates[1], "op3": candidates[2]})
            best_clean = clean_qwen_output(str(best_raw))
            match = re.search(r"\[?([123])\]?", best_clean)
            if match:
                candidate_num = int(match.group(1))
                print(f"   Juez eligió opción {candidate_num}")
            else:
                print("   Juez devolvió respuesta sin número explícito; usando como está")
                return best_clean

            parts = best_clean.split("\n", 1)
            if len(parts) > 1:
                routine = parts[1].strip()
            else:
                routine = best_clean.split("[" + str(candidate_num) + "]", 1)[-1].strip()
                if not routine:
                    routine = candidates[candidate_num - 1]
            return routine if routine else candidates[candidate_num - 1]
        except Exception as exc:
            print(f"⚠️ Error en judge_candidates: {exc}. Usando opción 1.")
            return candidates[0]

    def evaluate_rag_triad(self, question_spanish: str, answer: str, docs_retrieved: List[Document]) -> Dict[str, Any]:
        context_formatted = compress_context_for_triad(docs_retrieved) if docs_retrieved else "[Sin contexto]"

        if self._triad_chain is not None:
            try:
                evaluation_raw = safe_runnable_invoke(self._triad_chain, {"question_es": question_spanish, "context": context_formatted, "answer": answer})
                evaluation_clean = clean_qwen_output(str(evaluation_raw))
                evaluation_dict = extract_json_block(evaluation_clean)
                if isinstance(evaluation_dict, dict):
                    required_fields = {"relevancia_contexto", "fidelidad", "relevancia_respuesta", "score_general", "justificacion_contexto", "justificacion_fidelidad", "justificacion_respuesta", "recomendaciones"}
                    if not all(field in evaluation_dict for field in required_fields):
                        print("⚠️ JSON auditor incompleto; rellenando campos faltantes.")
                        defaults = {"relevancia_contexto": 3, "justificacion_contexto": "No disponible", "fidelidad": 3, "justificacion_fidelidad": "No disponible", "relevancia_respuesta": 3, "justificacion_respuesta": "No disponible", "score_general": 3, "recomendaciones": "Respuesta parcial del auditor"}
                        evaluation_dict = {**defaults, **evaluation_dict}
                    return evaluation_dict
                print(f"⚠️ No se pudo extraer JSON de respuesta auditor. Texto: {evaluation_clean[:200]}")
                print("⚠️ Usando evaluación heurística fallback.")
            except Exception as exc:
                print(f"⚠️ Error en evaluate_rag_triad (auditor LLM): {exc}. Usando evaluación heurística fallback.")

        try:
            def words(text: str) -> List[str]:
                return re.findall(r"\w+", text.lower())

            stopwords = {"el", "la", "los", "las", "y", "o", "de", "del", "que", "en", "por", "con", "a", "un", "una", "para", "se", "es", "su", "al", "lo", "como", "mas"}
            ctx_words = [word for word in words(context_formatted) if word not in stopwords]
            ans_words = [word for word in words(answer) if word not in stopwords]
            ques_words = [word for word in words(question_spanish) if word not in stopwords]
            ctx_set = set(ctx_words)
            match_ctx = sum(1 for word in ans_words if word in ctx_set)
            relevancia_contexto = min(5, max(1, int((match_ctx / max(1, min(60, len(ans_words)))) * 5)))
            non_ctx_in_answer = sum(1 for word in ans_words if word not in ctx_set)
            fidelity_ratio = 1 - (non_ctx_in_answer / max(1, len(ans_words)))
            fidelidad = min(5, max(1, int(fidelity_ratio * 5)))
            ques_set = set(ques_words)
            match_ques = sum(1 for word in ans_words if word in ques_set)
            relevancia_respuesta = min(5, max(1, int((match_ques / max(1, len(ques_words))) * 5)))
            score_general = round((relevancia_contexto + fidelidad + relevancia_respuesta) / 3)
            recommendations = []
            if relevancia_contexto <= 2:
                recommendations.append("Mejorar retrieval: aumentar docs/reducir ruido o activar resumen del contexto.")
            if fidelidad <= 2:
                recommendations.append("Revisar la respuesta: puede contener afirmaciones no soportadas por el contexto.")
            if relevancia_respuesta <= 2:
                recommendations.append("La respuesta puede no abordar directamente la pregunta; ajustar prompt o contexto.")
            return {
                "relevancia_contexto": relevancia_contexto,
                "justificacion_contexto": f"{match_ctx} tokens del contexto aparecen en la respuesta.",
                "fidelidad": fidelidad,
                "justificacion_fidelidad": f"{non_ctx_in_answer} tokens en la respuesta sin evidencia directa en el contexto.",
                "relevancia_respuesta": relevancia_respuesta,
                "justificacion_respuesta": f"{match_ques} tokens de la pregunta aparecen en la respuesta.",
                "score_general": score_general,
                "recomendaciones": " ".join(recommendations) if recommendations else "Sin observaciones",
            }
        except Exception as exc:
            print(f"⚠️ Error en evaluación heurística de Tríada: {exc}")
            return {"error": str(exc), "score_general": 0, "recomendaciones": "Evaluación no disponible"}

    def _summarize_plan(self, plan: List[Dict[str, Any]]) -> str:
        return json.dumps(plan, ensure_ascii=False, indent=2)

    def _synthesize_final_routine(self, state: Dict[str, Any]) -> str:
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            question_es=state.get("question_es", ""),
            question_en=state.get("question_en", ""),
            safety_warning=state.get("safety_warning", ""),
            days_per_week=state.get("days_per_week") or "No detectados",
            plan_summary=self._summarize_plan(state.get("plan", [])),
            task_results=trim_text(json.dumps(state.get("task_results", []), ensure_ascii=False, indent=2), 12000),
            scientific_context=trim_text(state.get("scientific_context", ""), 8000),
            wger_context=trim_text(state.get("wger_context", ""), 8000),
        )
        final_raw = safe_runnable_invoke(self._llm_synthesizer, prompt)
        final_text = final_raw.content if isinstance(final_raw, AIMessage) else str(final_raw)
        final_text = clean_qwen_output(final_text)
        safety_warning = state.get("safety_warning", "")
        if safety_warning and not final_text.startswith("**⚠️ ADVERTENCIA DE SEGURIDAD:**"):
            final_text = safety_warning + final_text
        return final_text

    def translate_question(self, question_spanish: str) -> str:
        translation_prompt = ChatPromptTemplate.from_template(
            """
            Traduce del español al inglés de forma literal y clara para búsqueda semántica.
            Devuelve solo la traducción final en inglés, sin explicaciones.

            Texto en español:
            {question_spanish}
            """
        )
        chain = translation_prompt | self._llm_translator | StrOutputParser()
        translated = safe_runnable_invoke(chain, {"question_spanish": question_spanish})
        return clean_qwen_output(str(translated))

    def retrieve_context(self, question_english: str) -> str:
        try:
            raw = self._consultar_rag_impl(question_english)
            return raw["context"]
        except Exception as exc:
            print(f"⚠️ Error en retrieve_context: {exc}")
            return "[Error: No se pudo recuperar contexto]"

    def generate_candidates(self, question_spanish: str, question_english: str, context: str, samples: int = 3) -> List[str]:
        responses: List[str] = []
        max_retries = 5
        for i in range(samples):
            print(f"   Generando opción {i + 1}/{samples}...")
            for attempt in range(max_retries):
                try:
                    raw_response = safe_runnable_invoke(self._generator_chain, {"context": context, "question_es": question_spanish, "question_en": question_english})
                    responses.append(clean_qwen_output(str(raw_response)))
                    break
                except Exception as exc:
                    error_msg = str(exc)
                    if detect_rate_limit_error(error_msg):
                        wait_time = 10 * (2 ** attempt) + random.uniform(0, 3)
                        print(f"   ⏳ Rate Limit detectado. Esperando {int(wait_time)}s (attempt {attempt + 1})...")
                        time.sleep(wait_time)
                    else:
                        wait_time = 2 * (attempt + 1) + random.uniform(0, 1)
                        print(f"   ⚠️ Error generando opción {i + 1} (attempt {attempt + 1}): {error_msg}. Esperando {int(wait_time)}s antes de reintentar.")
                        time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print(f"   ⚠️ Último intento fallido para opción {i + 1}: {error_msg}")
                        responses.append("[Error al generar rutina]")

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
        if len(candidates) < 3:
            if candidates:
                while len(candidates) < 3:
                    candidates.append(candidates[-1])
            else:
                return "[Error: no se generaron candidatos]"
        try:
            best_raw = safe_runnable_invoke(self._judge_chain, {"question_es": question_spanish, "op1": candidates[0], "op2": candidates[1], "op3": candidates[2]})
            best_clean = clean_qwen_output(str(best_raw))
            match = re.search(r"\[?([123])\]?", best_clean)
            if match:
                candidate_num = int(match.group(1))
                print(f"   Juez eligió opción {candidate_num}")
            else:
                print("   Juez devolvió respuesta sin número explícito; usando como está")
                return best_clean
            parts = best_clean.split("\n", 1)
            if len(parts) > 1:
                routine = parts[1].strip()
            else:
                routine = best_clean.split("[" + str(candidate_num) + "]", 1)[-1].strip()
                if not routine:
                    routine = candidates[candidate_num - 1]
            return routine if routine else candidates[candidate_num - 1]
        except Exception as exc:
            print(f"⚠️ Error en judge_candidates: {exc}. Usando opción 1.")
            return candidates[0]

    def run_planificador_orquestador(self, question_spanish: str) -> Dict[str, Any]:
        question_clean = question_spanish.strip()
        question_en = self.translate_question(question_clean)
        safety = detect_pain_signal(question_clean)
        safety_warning = build_safety_warning(question_clean)
        days_per_week = extract_days_per_week(question_clean)

        planner_raw = safe_runnable_invoke(self._llm_planner, self._plan_prompt(question_clean))
        planner_text = planner_raw.content if isinstance(planner_raw, AIMessage) else str(planner_raw)
        plan = extract_json_array_from_text(planner_text)
        if not plan:
            plan = [
                {"paso": 1, "tarea": "buscar evidencia científica relevante sobre volumen y progresión para principiante", "tool": "consultar_rag"},
                {"paso": 2, "tarea": "obtener ejercicios compatibles con el equipamiento disponible", "tool": "consultar_wger_api"},
                {"paso": 3, "tarea": "sintetizar una rutina segura y personalizada", "tool": "ninguna"},
            ]
        plan = sorted(plan, key=lambda item: item.get("paso", 999))

        state: Dict[str, Any] = {
            "question_es": question_clean,
            "question_en": question_en,
            "safety": safety,
            "safety_warning": safety_warning,
            "days_per_week": days_per_week,
            "plan": plan,
            "task_results": [],
            "execution_trace": [],
            "scientific_context": "",
            "wger_context": "",
            "equipment_text": question_clean,
        }

        for task in plan:
            step_result = self._invoke_executor_step(task, state)
            state["task_results"].append(step_result)
            state["execution_trace"].append(step_result)
            for tool_result in step_result.get("tool_results", []):
                tool_name = tool_result.get("tool")
                output = tool_result.get("output")
                if tool_name == "consultar_rag":
                    parsed = output if isinstance(output, dict) else extract_json_block(str(output))
                    if isinstance(parsed, dict):
                        context = parsed.get("context", "")
                        if context:
                            state["scientific_context"] = context
                elif tool_name == "consultar_wger_api":
                    parsed = output if isinstance(output, dict) else extract_json_block(str(output))
                    if isinstance(parsed, dict):
                        state["wger_context"] = trim_text(json.dumps(parsed, ensure_ascii=False, indent=2), 6000)

            if state.get("scientific_context") == "" and any(item.get("tool") == "consultar_rag" for item in plan):
                try:
                    fallback_rag = self._consultar_rag_impl(question_en)
                    state["scientific_context"] = fallback_rag.get("context", "")
                except Exception:
                    pass

            if state.get("wger_context") == "" and any(item.get("tool") == "consultar_wger_api" for item in plan):
                try:
                    fallback_wger = self._consultar_wger_api_impl(question_clean)
                    state["wger_context"] = trim_text(json.dumps(fallback_wger, ensure_ascii=False, indent=2), 6000)
                except Exception:
                    pass

            time.sleep(TASK_PAUSE_SECONDS)

        final_answer = self._synthesize_final_routine(state)

        docs_list: List[Document] = []
        try:
            docs_list = self._retriever.invoke(question_en) if hasattr(self._retriever, "invoke") else self._retriever_base.invoke(question_en)
        except Exception:
            docs_list = []

        result = {
            "question_spanish": question_clean,
            "question_english": question_en,
            "question_en": question_en,
            "plan": plan,
            "task_results": state["task_results"],
            "execution_trace": state["execution_trace"],
            "scientific_context": state["scientific_context"],
            "wger_context": state["wger_context"],
            "safety_warning": safety_warning,
            "final_answer": final_answer,
            "context": state["scientific_context"],
            "candidates": [item.get("assistant_summary", "") for item in state["task_results"]],
        }

        result["rag_triad_evaluation"] = self.evaluate_rag_triad(
            question_spanish=question_clean,
            answer=final_answer,
            docs_retrieved=docs_list,
        )
        return result

    def run_pipeline(self, question_spanish: str, samples: int = 3):
        orchestrated = self.run_planificador_orquestador(question_spanish)
        orchestrated["samples"] = samples
        return orchestrated

    def generate_routine(self, question_spanish: str, samples: int = 3) -> str:
        return self.run_pipeline(question_spanish, samples=samples)["final_answer"]

    def run_interactive_console(self, question_spanish: str, samples: int = 3) -> str:
        print("[Paso 1] Modo planificador: generando subtareas en JSON...")
        orchestrated = self.run_planificador_orquestador(question_spanish)
        print("\nPlan generado:")
        print(json.dumps(orchestrated["plan"], ensure_ascii=False, indent=2))
        wait_for_continue("\nPresiona una tecla para continuar al final de la ejecución...")
        print("\nRespuesta final:")
        print(orchestrated["final_answer"])
        return orchestrated["final_answer"]


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


def build_routine_result(question_usuario: str, chroma_path: str = CHROMA_PATH, samples: int = 3) -> str:
    agent = RoutineRAGAgent(chroma_path=chroma_path)
    return agent.generate_routine(question_usuario, samples=samples)


def generar_rutina_robusta(pregunta_usuario: str, chroma_path: str = CHROMA_PATH, samples: int = 3) -> str:
    return build_routine_result(pregunta_usuario, chroma_path=chroma_path, samples=samples)


if __name__ == "__main__":
    pregunta = """
    Hola, soy principiante y quiero aumentar la masa muscular de mi tren superior y piernas
    entrenando en mi casa 3 veces por semana. Estrictamente NO tengo pesas ni mancuernas,
    solo puedo usar mi peso corporal (calistenia) y una silla.
    Además, me duele un poco la rodilla derecha al flectarla mucho.
    ¿Me puedes armar una rutina y explicarme cuántas series y descansos necesito según la ciencia?
    """
    print(generar_rutina_robusta(pregunta))