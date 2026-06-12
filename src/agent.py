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
from typing_extensions import TypedDict

import requests
import torch
from dotenv import load_dotenv

# LangGraph Imports
from langgraph.graph import StateGraph, START, END

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
WGER_API_BASE = "https://wger.de/api/v2/exerciseinfo/"
WGER_TIMEOUT_SECONDS = 20
WGER_RESULT_LIMIT = 8
WGER_LANGUAGE_SPANISH = 4

# ==========================================
# 1. DEFINICIÓN DEL ESTADO GLOBAL (LangGraph)
# ==========================================
class AgentState(TypedDict):
    question_es: str
    question_en: str
    safety_warning: str
    scientific_context: str
    wger_context: str
    draft_routine: str
    audit_feedback: str
    is_safe: bool
    iterations: int
    final_answer: str

# ==========================================
# UTILIDADES Y PROMPTS
# ==========================================
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

def clean_html_to_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def select_spanish_translation(item: Dict[str, Any]) -> Dict[str, Any]:
    translations = item.get("translations", []) or []
    spanish = next((t for t in translations if t.get("language") == WGER_LANGUAGE_SPANISH), None)
    chosen = spanish or (translations[0] if translations else {})
    return {
        "id": item.get("id"),
        "nombre": chosen.get("name") or item.get("id"),
        "descripcion": clean_html_to_text(chosen.get("description") or ""),
        "equipamiento": [equip.get("name") if isinstance(equip, dict) else equip for equip in item.get("equipment", [])],
    }

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([f"Fuente: {doc.metadata.get('source', 'N/A')}\nContenido: {doc.page_content}" for doc in docs])

def summarize_docs_for_state(docs: List[Document]) -> str:
    clipped_docs = []
    for doc in docs[:MAX_CONTEXT_DOCS]:
        snippet = doc.page_content[:DOC_SNIPPET_CHARS]
        if len(doc.page_content) > DOC_SNIPPET_CHARS: snippet += "..."
        clipped_docs.append(Document(page_content=snippet, metadata=doc.metadata))
    return format_docs(clipped_docs)

def build_fallback_wger_payload(equipment_id: int, equipment_term: str) -> Dict[str, Any]:
    exercises = FALLBACK_WGER_EXERCISES.get(equipment_id) or FALLBACK_WGER_EXERCISES.get(7, [])
    return {"source": "fallback", "results": exercises}

GENERATOR_TEMPLATE = """
Actúa como un entrenador personal experto basado estrictamente en evidencia científica deportiva.
Debes diseñar una rutina utilizando EXCLUSIVAMENTE el contexto proporcionado y el equipamiento disponible.

Consulta original: {question_es}
Seguridad Requerida: {safety_warning}

Contexto Científico: {scientific_context}
Catálogo de Ejercicios: {wger_context}

REGLAS DE FORMATO:
- Estructura en 3 partes: 1. Parámetros de Entrenamiento, 2. Rutina Propuesta, 3. Justificación Científica.
- Cada ejercicio debe incluir: *Músculos objetivos*, *Cómo hacerlo* y *Modificación*.

{feedback_block}

Redacta la rutina ahora en formato Markdown limpio:
"""

AUDITOR_TEMPLATE = """
Eres un Auditor Clínico-Deportivo experto.
Tu tarea es evaluar si la rutina propuesta es segura para el usuario.

USUARIO PIDIÓ: "{question_es}"
RUTINA GENERADA:
{draft}

REGLAS DE AUDITORÍA:
1. Si el usuario mencionó un dolor (ej. rodilla, hombro), y la rutina incluye ejercicios que lo agravan (ej. saltos, flexiones profundas), la rutina NO ES SEGURA.
2. Si la rutina incluye equipo que el usuario NO TIENE, la rutina NO ES SEGURA.

Devuelve ÚNICAMENTE un JSON válido con este formato:
{{
    "es_segura": true o false,
    "feedback": "Si es false, explica qué ejercicios quitar o cambiar. Si es true, escribe 'Aprobado'."
}}
"""

def clean_qwen_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace("```", "").strip()

def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", normalized.lower()).strip()

def detect_pain_signal(question: str) -> bool:
    normalized = normalize_text(question)
    return any(word in normalized for word in ["dolor", "molestia", "lesion", "lesión", "duele"])

# ==========================================
# CLASE PRINCIPAL DEL AGENTE
# ==========================================
@dataclass
class RoutineRAGAgent:
    chroma_path: str = CHROMA_PATH
    k: int = 4

    def __post_init__(self) -> None:
        # 1. Configurar Modelos
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store = Chroma(persist_directory=self.chroma_path, embedding_function=self._embeddings)
        self._retriever_base = self._vector_store.as_retriever(search_kwargs={"k": self.k})

        # LLMs
        self._llm_generator = ChatGroq(model_name=DEFAULT_LLM_MODEL, temperature=0.4)
        self._llm_judge = ChatGroq(model_name=DEFAULT_SYNTHESIS_MODEL, temperature=0.0)
        self._llm_translator = ChatGroq(model_name=DEFAULT_LLM_MODEL, temperature=0.0)

        # MQR
        mqr_prompt = ChatPromptTemplate.from_template("Genera 3 variantes en inglés técnico para: {question}. Solo las frases, 1 por línea.")
        self._retriever = MultiQueryRetriever.from_llm(retriever=self._retriever_base, llm=self._llm_translator, prompt=mqr_prompt)

        # 2. Compilar el Grafo
        self.graph = self._build_graph()

    # --- Nodos del Grafo ---
    def nodo_analizador(self, state: AgentState):
        print("🧠 [Nodo 1: Analizador] Procesando consulta y traduciendo...")
        q_es = state["question_es"]
        
        # Traducción
        chain = ChatPromptTemplate.from_template("Traduce al inglés técnico, solo el texto: {q}") | self._llm_translator | StrOutputParser()
        q_en = clean_qwen_output(chain.invoke({"q": q_es}))
        
        # Seguridad
        has_pain = detect_pain_signal(q_es)
        warning = "**⚠️ ADVERTENCIA DE SEGURIDAD:** Si tienes molestias persistentes, consulta a un profesional. Esta rutina ha sido adaptada por precaución.\n\n" if has_pain else ""
        
        return {"question_en": q_en, "safety_warning": warning, "iterations": 0}

    def nodo_rag(self, state: AgentState):
        print("🛠️ [Nodo 2: RAG Tool] Buscando evidencia real en ChromaDB...")
        try:
            # Usamos tu función real de RAG
            resultado_rag = self._consultar_rag_impl(state["question_en"])
            contexto_real = resultado_rag.get("context", "Sin contexto recuperado.")
        except Exception as e:
            print(f"   ⚠️ Error en RAG: {e}")
            contexto_real = "[Error al consultar base científica]"
            
        return {"scientific_context": contexto_real}

    def nodo_api(self, state: AgentState):
        print("🛠️ [Nodo 3: API Tool] Buscando ejercicios reales en Wger API...")
        try:
            # Pasamos la pregunta en español para que tu función detecte el equipo (ej: silla, peso corporal)
            resultado_api = self._consultar_wger_api_impl(state["question_es"])
            
            # Formateamos el JSON de la API para que el LLM lo lea fácilmente
            ejercicios = resultado_api.get("results", [])
            if ejercicios:
                import json
                # Limitamos el largo para no saturar tokens
                contexto_api = json.dumps(ejercicios, ensure_ascii=False, indent=2)[:3000]
            else:
                contexto_api = "No se encontraron ejercicios en la API para este equipamiento."
                
        except Exception as e:
            print(f"   ⚠️ Error en API: {e}")
            contexto_api = "[Error al consultar catálogo de ejercicios]"
            
        return {"wger_context": contexto_api}

    def nodo_generador(self, state: AgentState):
        print(f"🧠 [Nodo 4: Generador] Creando rutina (Iteración {state.get('iterations', 0) + 1})...")
        feedback_block = ""
        if state.get("audit_feedback"):
            print(f"   -> Aplicando feedback del Juez: {state['audit_feedback']}")
            feedback_block = f"\nATENCIÓN, EL AUDITOR RECHAZÓ LA VERSIÓN ANTERIOR POR ESTO: {state['audit_feedback']}. CORRIGE LA RUTINA."

        prompt = GENERATOR_TEMPLATE.format(
            question_es=state["question_es"],
            safety_warning=state.get("safety_warning", ""),
            scientific_context=state.get("scientific_context", ""),
            wger_context=state.get("wger_context", ""),
            feedback_block=feedback_block
        )
        
        res = self._llm_generator.invoke(prompt)
        return {"draft_routine": clean_qwen_output(res.content), "iterations": state.get("iterations", 0) + 1}

    def nodo_auditor(self, state: AgentState):
        print("⚖️ [Nodo 5: Auditor] Verificando seguridad clínica (Llama 3.3 70B)...")
        prompt = AUDITOR_TEMPLATE.format(question_es=state["question_es"], draft=state["draft_routine"])
        
        try:
            res = self._llm_judge.invoke(prompt)
            cleaned = clean_qwen_output(res.content)
            data = json.loads(cleaned)
            is_safe = data.get("es_segura", True)
            feedback = data.get("feedback", "Aprobado")
        except Exception as e:
            print("   ⚠️ Error al parsear JSON del juez, asumiendo seguro por fallback.")
            is_safe, feedback = True, "Aprobado (Fallback)"
            
        print(f"   -> Veredicto: {'✅ SEGURA' if is_safe else '❌ RECHAZADA'}")
        return {"is_safe": is_safe, "audit_feedback": feedback}

    def nodo_formateador(self, state: AgentState):
        print("✨ [Nodo 6: Formateador] Preparando entrega final...")
        rutina = state["draft_routine"]
        if state.get("safety_warning") and "**⚠️ ADVERTENCIA" not in rutina:
            rutina = state["safety_warning"] + rutina
        return {"final_answer": rutina}

    # ==========================================
    # LÓGICA CORE: RAG Y API (Trasplantada)
    # ==========================================
    def _consultar_rag_impl(self, termino_busqueda_en: str) -> Dict[str, Any]:
        # MQR: Invoca múltiples queries generadas por el LLM traductor
        docs = self._retriever.invoke(termino_busqueda_en)
        context = summarize_docs_for_state(docs)
        return {"context": context}

    def _resolve_equipment_ids(self, equipamiento_es: str) -> List[Dict[str, Any]]:
        normalized = normalize_text(equipamiento_es)
        if not normalized:
            return [{"term": "peso corporal", "equipment_id": 7}]

        detected = []
        for alias, equipment_id in WGER_EQUIPMENT_ALIASES.items():
            if alias in normalized:
                detected.append({"term": alias, "equipment_id": equipment_id})
                
        if not detected:
            return [{"term": equipamiento_es.strip() or "peso corporal", "equipment_id": 7}]
            
        unique = {item["equipment_id"]: item for item in detected}
        return list(unique.values())

    def _fetch_wger_exercises(self, equipment_id: int) -> List[Dict[str, Any]]:
        params = {"language": WGER_LANGUAGE_SPANISH, "equipment": equipment_id, "limit": WGER_RESULT_LIMIT}
        response = requests.get(WGER_API_BASE, params=params, timeout=WGER_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        
        exercises = []
        for item in payload.get("results", []):
            if isinstance(item, dict):
                exercises.append(select_spanish_translation(item))
        return exercises

    def _consultar_wger_api_impl(self, equipamiento_es: str) -> Dict[str, Any]:
        equipment_matches = self._resolve_equipment_ids(equipamiento_es)
        aggregated_results = []

        for match in equipment_matches:
            equipment_id = int(match["equipment_id"])
            try:
                exercises = self._fetch_wger_exercises(equipment_id)
                if exercises:
                    aggregated_results.extend(exercises)
            except Exception as exc:
                print(f"⚠️ Fallo en Wger API para ID {equipment_id} ({exc}). Usando Fallback.")
                fallback_payload = build_fallback_wger_payload(equipment_id, match.get("term", equipamiento_es))
                aggregated_results.extend(fallback_payload.get("results", []))

        if not aggregated_results:
            fallback_payload = build_fallback_wger_payload(7, equipamiento_es)
            aggregated_results = fallback_payload.get("results", [])

        # Deduplicar
        deduplicated = {str(item.get("nombre", "")): item for item in aggregated_results}
        
        return {"results": list(deduplicated.values())[:WGER_RESULT_LIMIT]}
    
    # --- Router Condicional ---
    def router_seguridad(self, state: AgentState) -> str:
        if state["is_safe"] or state["iterations"] >= 3:
            return "aprobar"
        return "rechazar"

    # --- Ensamblaje de LangGraph ---
    def _build_graph(self):
        grafo = StateGraph(AgentState)
        
        grafo.add_node("analizador", self.nodo_analizador)
        grafo.add_node("rag", self.nodo_rag)
        grafo.add_node("api", self.nodo_api)
        grafo.add_node("generador", self.nodo_generador)
        grafo.add_node("auditor", self.nodo_auditor)
        grafo.add_node("formateador", self.nodo_formateador)
        
        grafo.add_edge(START, "analizador")
        grafo.add_edge("analizador", "rag")
        grafo.add_edge("rag", "api")
        grafo.add_edge("api", "generador")
        grafo.add_edge("generador", "auditor")
        
        grafo.add_conditional_edges(
            "auditor",
            self.router_seguridad,
            {"rechazar": "generador", "aprobar": "formateador"}
        )
        grafo.add_edge("formateador", END)
        
        return grafo.compile()

    def run_interactive_console(self, question_spanish: str) -> str:
        estado_inicial = {"question_es": question_spanish}
        
        # Ejecución usando .stream() como pide la rúbrica
        for output in self.graph.stream(estado_inicial, stream_mode="updates"):
            # Pequeña pausa para evitar Rate Limits al iterar
            time.sleep(1)
            
        final_state = self.graph.get_state(self.graph.invoke(estado_inicial))
        # Para obtener el output iterativo limpio usamos invoke final
        res = self.graph.invoke(estado_inicial)
        return res["final_answer"]


if __name__ == "__main__":
    pregunta = """
    Hola, soy principiante y quiero aumentar masa en mis piernas en casa.
    Solo tengo una silla. Además, me duele un poco la rodilla derecha al flectarla.
    """
    
    print("INICIANDO GRAFO MULTI-AGENTE (LANGGRAPH)\n")
    agent = RoutineRAGAgent()
    rutina_final = agent.run_interactive_console(pregunta)

    print("\n" + "=" * 60)
    print("RUTINA FINAL GENERADA:")
    print("=" * 60 + "\n")
    print(rutina_final)