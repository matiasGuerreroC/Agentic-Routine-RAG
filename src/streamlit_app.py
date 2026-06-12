import os
import json
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
    .node-header {
        border-left: 5px solid #2f6f4e;
        background: #effaf2;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
    }
    .tool-header {
        border-left: 5px solid #c27c2f;
        background: #fff8ef;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
    }
    .judge-header {
        border-left: 5px solid #9c27b0;
        background: #f3e5f5;
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
        "final_state": None,
        "graph_events": [], # Guardaremos los eventos del grafo aquí
        "last_question": "",
        "loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_panel(title: str, content: str) -> None:
    st.markdown(f'<div class="panel-card"><strong>{title}</strong></div>', unsafe_allow_html=True)
    st.markdown(content)

def run_langgraph(agent: RoutineRAGAgent, question: str):
    """Ejecuta el grafo y captura los eventos nodo por nodo sin ejecutar dos veces"""
    estado_inicial = {"question_es": question}
    eventos = []
    
    # Creamos un diccionario para ir acumulando el estado final
    final_state = dict(estado_inicial)
    
    # Capturamos el flujo del grafo en tiempo real
    for event in agent.graph.stream(estado_inicial, stream_mode="updates"):
        eventos.append(event)
        # LangGraph nos da "deltas" (cambios). Los sumamos al estado final.
        for node_name, node_output in event.items():
            final_state.update(node_output)
            
    return final_state, eventos

def main() -> None:
    init_state()
    agent = get_agent()

    st.title("Agentic Routine RAG (LangGraph)")
    st.write(
        "Esta interfaz muestra el flujo real del **Grafo de Estados (Reflexion)**: análisis, "
        "llamada a herramientas (RAG/API), generación y evaluación cíclica del Juez."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning("No se detectó GROQ_API_KEY en el entorno. Configúrela antes de ejecutar.")

    with st.sidebar:
        st.header("Configuración de Nodos")
        st.write("Generador/Planificador: qwen/qwen3-32b")
        st.write("Juez/Auditor: llama-3.3-70b-versatile")
        st.write("Embedding: nomic-ai/nomic-embed-text-v1.5")
        st.markdown("---")
        st.info("El panel inferior mostrará cómo LangGraph enruta el estado entre los diferentes nodos (analizador, rag, api, generador, auditor).")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Pregunta del usuario en español",
            height=120,
            value=st.session_state.question_input,
            placeholder="Ej: Quiero hipertrofia en piernas. No tengo equipo. Me duele la espalda baja.",
        )
        submitted = st.form_submit_button("Ejecutar Grafo de Agentes", use_container_width=True)

    if submitted:
        st.session_state.question_input = question
        st.session_state.last_question = question.strip()
        st.session_state.loading = True
        
        with st.spinner("LangGraph está orquestando los nodos..."):
            final_state, events = run_langgraph(agent, question)
            st.session_state.final_state = final_state 
            st.session_state.graph_events = events
            
        st.session_state.loading = False
        st.rerun()

    if not st.session_state.last_question:
        st.info("Escribe una petición y presiona 'Ejecutar Grafo de Agentes' para iniciar.")
        return

    state = st.session_state.final_state
    events = st.session_state.graph_events
    
    if not state:
        st.info("Aún no hay ejecución guardada.")
        return

    # --- MÉTRICAS SUPERIORES ---
    progress_cols = st.columns(4)
    progress_cols[0].metric("Iteraciones de Generación", state.get("iterations", 1))
    progress_cols[1].metric("Evaluación del Juez", "Segura ✅" if state.get("is_safe") else "Insegura ❌")
    progress_cols[2].metric("Herramienta RAG", "Ejecutada")
    progress_cols[3].metric("Herramienta API", "Ejecutada")

    st.markdown("---")

    # --- ANÁLISIS DE ENTRADA ---
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Entrada y Perfilamiento")
        render_panel("Pregunta Original (ES)", st.session_state.last_question)
        st.markdown(f"**Traducción MQR (EN):** `{state.get('question_en', '')}`")
        if state.get("safety_warning"):
            st.warning("⚠️ **Alerta de Salud Detectada por el Analizador**")

    with right:
        st.subheader("Auditoría Final (Juez Llama 70B)")
        st.caption("Feedback interno generado por el nodo 'auditor' antes del renderizado final.")
        if state.get("is_safe"):
            st.success(state.get("audit_feedback", "Aprobado por el juez."))
        else:
            st.error(f"Rechazado en iteración. Feedback: {state.get('audit_feedback', '')}")

    st.markdown("---")
    
    # --- TRAZA DEL GRAFO (LANGGRAPH) ---
    st.subheader("Traza de Ejecución (Nodos de LangGraph)")
    st.caption("Cada bloque muestra la mutación del Estado Global (State) al pasar por un nodo.")
    
    for idx, event in enumerate(events, 1):
        for node_name, node_output in event.items():
            if node_name == "analizador":
                st.markdown(f'<div class="node-header"><strong>Nodo {idx}: Analizador</strong></div>', unsafe_allow_html=True)
                st.write("El agente tradujo la consulta y detectó las alertas de seguridad.")
            
            elif node_name == "rag":
                st.markdown(f'<div class="tool-header"><strong>Nodo {idx}: Herramienta RAG (ChromaDB)</strong></div>', unsafe_allow_html=True)
                with st.expander("Ver Contexto Científico Recuperado"):
                    st.text_area("Contexto Científico", value=node_output.get("scientific_context", ""), height=150, disabled=True, label_visibility="collapsed")
            
            elif node_name == "api":
                st.markdown(f'<div class="tool-header"><strong>Nodo {idx}: Herramienta API (Wger/Fallback)</strong></div>', unsafe_allow_html=True)
                with st.expander("Ver Catálogo de Ejercicios Recuperado"):
                    st.text_area("Catálogo de Ejercicios (API Wger)", value=node_output.get("wger_context", ""), height=150, disabled=True, label_visibility="collapsed")
            
            elif node_name == "generador":
                st.markdown(f'<div class="node-header"><strong>Nodo {idx}: Generador (Iteración {node_output.get("iterations", 1)})</strong></div>', unsafe_allow_html=True)
                with st.expander("Ver Borrador de la Rutina"):
                    st.markdown(node_output.get("draft_routine", ""))
                    
            elif node_name == "auditor":
                st.markdown(f'<div class="judge-header"><strong>⚖️ Nodo {idx}: Auditor de Seguridad Clínica</strong></div>', unsafe_allow_html=True)
                st.write(f"**Veredicto:** {'Aprobado ✅' if node_output.get('is_safe') else 'Rechazado ❌'}")
                st.write(f"**Feedback inyectado al estado:** {node_output.get('audit_feedback')}")

    st.markdown("---")
    
    # --- RESULTADO FINAL ---
    st.subheader("Rutina Final Entregada")
    st.markdown(state.get("final_answer", ""))

    st.download_button(
        "Descargar rutina en Markdown",
        data=state.get("final_answer", ""),
        file_name="rutina_final.md",
        mime="text/markdown",
        use_container_width=True,
    )

if __name__ == "__main__":
    main()