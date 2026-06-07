import json
import os

import torch
import streamlit as st

from agent import CHROMA_PATH, RoutineRAGAgent


st.set_page_config(
    page_title="Agentic Routine RAG",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .panel-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(248,249,250,0.82));
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .trace-call {
        border-left: 5px solid #c27c2f;
        background: #fff8ef;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
    }
    .trace-message {
        border-left: 5px solid #1f7a8c;
        background: #eef8fb;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
    }
    .trace-assistant {
        border-left: 5px solid #2f6f4e;
        background: #effaf2;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent() -> RoutineRAGAgent:
    return RoutineRAGAgent(chroma_path=CHROMA_PATH)


def init_state() -> None:
    defaults = {
        "question_input": "",
        "result": None,
        "last_question": "",
        "loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_panel(title: str, content: str) -> None:
    st.markdown(f'<div class="panel-card"><strong>{title}</strong></div>', unsafe_allow_html=True)
    st.markdown(content)


def render_trace_event(event: dict, index: int) -> None:
    event_type = event.get("type", "evento")
    if event_type == "tool_call":
        st.markdown(f'<div class="trace-call"><strong>tool_call #{index}</strong></div>', unsafe_allow_html=True)
        st.markdown(f"**Herramienta:** `{event.get('tool', '')}`")
        st.markdown("**Argumentos estructurados:**")
        st.code(json.dumps(event.get("arguments", {}), ensure_ascii=False, indent=2), language="json")
    elif event_type == "tool_message":
        st.markdown(f'<div class="trace-message"><strong>tool_message #{index}</strong></div>', unsafe_allow_html=True)
        st.markdown(f"**Herramienta:** `{event.get('tool', '')}`")
        payload = event.get("content", "")
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                st.json(parsed)
            except Exception:
                st.code(payload, language="json")
        else:
            st.json(payload)
    else:
        st.markdown(f'<div class="trace-assistant"><strong>{event_type}</strong></div>', unsafe_allow_html=True)
        st.write(event.get("content", ""))


def run_orchestrator(agent: RoutineRAGAgent, question: str) -> dict:
    with st.spinner("Ejecutando planificador, herramientas y síntesis final..."):
        return agent.run_planificador_orquestador(question)


def main() -> None:
    init_state()
    agent = get_agent()

    st.title("Agentic Routine RAG")
    st.write(
        "Esta interfaz muestra el flujo real del orquestador: planificación, decisión de herramienta, llamada a la API y devolución del resultado al LLM."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning("No se detectó GROQ_API_KEY en el entorno. El agente no podrá generar respuestas hasta configurarla.")

    with st.sidebar:
        st.header("Configuración")
        st.write("Modelo LLM: qwen/qwen3-32b")
        st.write("Embedding: nomic-ai/nomic-embed-text-v1.5")
        st.write(f"Base vectorial: {CHROMA_PATH}")
        st.markdown("---")
        st.info(
            "El panel principal muestra el momento exacto en que el LLM emite `tool_call`, los argumentos enviados y el `tool_message` que vuelve al modelo."
        )

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Pregunta del usuario en español",
            height=180,
            value=st.session_state.question_input,
            placeholder="Describe tu objetivo, equipo disponible, días de entrenamiento y molestias si las hay.",
        )
        submitted = st.form_submit_button("Ejecutar orquestador", use_container_width=True)

    if submitted:
        st.session_state.question_input = question
        st.session_state.last_question = question.strip()
        st.session_state.loading = True
        st.session_state.result = run_orchestrator(agent, question)
        st.session_state.loading = False
        st.rerun()

    if not st.session_state.last_question:
        st.info("Escribe una pregunta y presiona 'Ejecutar orquestador' para ver el flujo completo.")
        return

    result = st.session_state.result
    if not result:
        st.info("Aún no hay una ejecución guardada.")
        return

    progress_cols = st.columns(4)
    progress_cols[0].metric("Plan", len(result.get("plan", [])))
    progress_cols[1].metric("Pasos ejecutados", len(result.get("execution_trace", [])))
    progress_cols[2].metric("Herramienta RAG", "sí" if result.get("scientific_context") else "no")
    progress_cols[3].metric("Herramienta Wger", "sí" if result.get("wger_context") else "no")

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Entrada")
        render_panel("Pregunta en español", st.session_state.last_question)
        st.markdown(f"**Traducción interna:** {result.get('question_english', '')}")
        if result.get("safety_warning"):
            st.warning(result["safety_warning"])

    with right:
        st.subheader("Plan del LLM")
        st.caption("El planificador devuelve subtareas en JSON antes de ejecutar herramientas.")
        st.code(json.dumps(result.get("plan", []), ensure_ascii=False, indent=2), language="json")

    st.markdown("---")
    st.subheader("Flujo real de comunicación")
    st.caption("Cada bloque muestra la subtarea, el tool_call emitido por el LLM, la ejecución Python y el tool_message devuelto al modelo.")

    trace = result.get("execution_trace", [])
    if not trace:
        st.info("No se registraron pasos de ejecución.")
    else:
        for step_index, step in enumerate(trace, start=1):
            task = step.get("task", {})
            title = f"Paso {task.get('paso', step_index)} - {task.get('tarea', 'Subtarea')}"
            with st.expander(title, expanded=step_index == 1):
                st.markdown("**Subtarea planificada:**")
                st.json(task)
                st.markdown("**Resumen del ejecutor:**")
                st.write(step.get("assistant_summary", ""))

                events = step.get("trace_events", []) or []
                if events:
                    st.markdown("**Trazas de herramienta y respuesta:**")
                    for event_index, event in enumerate(events, start=1):
                        render_trace_event(event, event_index)
                else:
                    st.info("Este paso no requirió herramienta explícita.")

    st.markdown("---")
    st.subheader("Contexto recuperado")
    context_cols = st.columns(2)
    with context_cols[0]:
        st.markdown("**RAG científico**")
        st.text_area("", value=result.get("scientific_context", ""), height=260, disabled=True, label_visibility="collapsed")
    with context_cols[1]:
        st.markdown("**Wger API**")
        st.text_area("", value=result.get("wger_context", ""), height=260, disabled=True, label_visibility="collapsed")

    st.markdown("---")
    st.subheader("Rutina final")
    st.markdown(result.get("final_answer", ""))

    if result.get("rag_triad_evaluation"):
        st.markdown("---")
        st.subheader("Evaluación de calidad RAG")
        eval_data = result["rag_triad_evaluation"]
        metric_cols = st.columns(3)

        with metric_cols[0]:
            rel_contexto = eval_data.get("relevancia_contexto", 0)
            st.metric("Relevancia Contexto", f"{rel_contexto}/5")
            st.caption(eval_data.get("justificacion_contexto", ""))

        with metric_cols[1]:
            fidelidad = eval_data.get("fidelidad", 0)
            st.metric("Fidelidad", f"{fidelidad}/5")
            st.caption(eval_data.get("justificacion_fidelidad", ""))

        with metric_cols[2]:
            rel_respuesta = eval_data.get("relevancia_respuesta", 0)
            st.metric("Relevancia Respuesta", f"{rel_respuesta}/5")
            st.caption(eval_data.get("justificacion_respuesta", ""))

        st.metric("Score General RAG", f"{eval_data.get('score_general', 0)}/5")
        if eval_data.get("recomendaciones"):
            st.info(f"Observaciones: {eval_data['recomendaciones']}")

    st.download_button(
        "Descargar respuesta en Markdown",
        data=result.get("final_answer", ""),
        file_name="rutina_final.md",
        mime="text/markdown",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()