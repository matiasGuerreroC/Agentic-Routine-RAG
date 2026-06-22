import os
import json
import time
import streamlit as st

from agent import CHROMA_PATH, RoutineRAGAgent

st.set_page_config(
    page_title="Agentic Routine RAG",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS PERSONALIZADO ---
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
        animation: fadeIn 0.5s;
    }
    .tool-header {
        border-left: 5px solid #c27c2f;
        background: #fff8ef;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.5s;
    }
    .judge-header {
        border-left: 5px solid #9c27b0;
        background: #f3e5f5;
        padding: 0.9rem 1rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
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
        "graph_events": [],
        "last_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main() -> None:
    init_state()
    agent = get_agent()

    st.title("Agentic Routine RAG (LangGraph)")
    st.write("Demostración en vivo del **Patrón de Reflexión**: Ejecución de herramientas, generación y auditoría cíclica.")

    if not os.getenv("GROQ_API_KEY"):
        st.warning("No se detectó GROQ_API_KEY en el entorno. Configúrela antes de ejecutar.")

    with st.sidebar:
        st.header("Configuración de Nodos")
        st.write("Generador/Planificador: qwen/qwen3-32b")
        st.write("Juez/Auditor: llama-3.3-70b-versatile")
        st.write("Embedding: nomic-ai/nomic-embed-text-v1.5")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Pregunta del usuario en español",
            height=100,
            value=st.session_state.question_input,
            placeholder="Ej: Quiero hipertrofia en pecho. Obligatoriamente flexiones, pero tengo dolor de hombro.",
        )
        submitted = st.form_submit_button("Ejecutar Grafo de Agentes", use_container_width=True)

    # ==========================================
    # ZONA DE EJECUCIÓN DINÁMICA (EFECTO CASCADA)
    # ==========================================
    if submitted and question.strip():
        st.session_state.last_question = question.strip()
        st.session_state.graph_events = []
        
        estado_inicial = {"question_es": question.strip(), "iterations": 0}
        final_state = dict(estado_inicial)

        st.markdown("---")
        
        # 1. Creamos "Contenedores Vacíos" que iremos actualizando en vivo
        metrics_placeholder = st.empty()
        st.subheader("Traza de Ejecución en Vivo")
        trace_container = st.container()
        final_result_placeholder = st.empty()

        # 2. Iteramos el grafo en tiempo real
        idx = 1
        for event in agent.graph.stream(estado_inicial, stream_mode="updates"):
            st.session_state.graph_events.append(event)
            
            for node_name, node_output in event.items():
                final_state.update(node_output)
                
                # A) Actualizamos los contadores de arriba (Métricas) en vivo
                with metrics_placeholder.container():
                    cols = st.columns(4)
                    cols[0].metric("Iteraciones", final_state.get("iterations", 0))
                    
                    is_safe = final_state.get("is_safe")
                    if is_safe is None:
                        juez_txt = "⏳ Evaluando..."
                    else:
                        juez_txt = "Segura ✅" if is_safe else "Insegura ❌"
                    cols[1].metric("Veredicto Juez", juez_txt)
                    
                    cols[2].metric("Herramienta RAG", "Ejecutada 🔍" if "scientific_context" in final_state else "Esperando...")
                    cols[3].metric("Herramienta API", "Ejecutada 🏋️" if "wger_context" in final_state else "Esperando...")

                # B) Imprimimos el nodo en cascada
                with trace_container:
                    if node_name == "analizador":
                        st.markdown(f'<div class="node-header"><strong>🧠 Nodo {idx}: Analizador</strong></div>', unsafe_allow_html=True)
                        st.info(f"**Traducción MQR (EN):** {node_output.get('question_en', '')}")
                        if node_output.get("safety_warning"):
                            st.warning("⚠️ Alerta Clínica Detectada")
                            
                    elif node_name == "rag":
                        st.markdown(f'<div class="tool-header"><strong>📚 Nodo {idx}: Herramienta RAG (ChromaDB)</strong></div>', unsafe_allow_html=True)
                        with st.expander("Ver Contexto Científico Recuperado"):
                            st.caption(node_output.get("scientific_context", ""))
                            
                    elif node_name == "api":
                        st.markdown(f'<div class="tool-header"><strong>🔌 Nodo {idx}: Herramienta API (Wger/Fallback)</strong></div>', unsafe_allow_html=True)
                        with st.expander("Ver Catálogo de Ejercicios"):
                            st.caption(node_output.get("wger_context", ""))
                            
                    elif node_name == "generador":
                        st.markdown(f'<div class="node-header"><strong>⚡ Nodo {idx}: Generador (Iteración {final_state.get("iterations", 1)})</strong></div>', unsafe_allow_html=True)
                        with st.expander("Ver Borrador de la Rutina (Qwen)"):
                            st.markdown(node_output.get("draft_routine", "Rutina generada."))
                            
                    elif node_name == "auditor":
                        st.markdown(f'<div class="judge-header"><strong>⚖️ Nodo {idx}: Auditor de Seguridad (Llama 70B)</strong></div>', unsafe_allow_html=True)
                        if node_output.get("is_safe"):
                            st.success("✅ Veredicto: Rutina Aprobada.")
                        else:
                            st.error(f"❌ Veredicto: RECHAZADA. \n\n**Feedback inyectado:** {node_output.get('audit_feedback', '')}")
                    
                    elif node_name == "formateador":
                        st.markdown(f'<div class="node-header"><strong>✨ Nodo {idx}: Formateador</strong></div>', unsafe_allow_html=True)
                        st.write("Ensamblando entrega final...")

                idx += 1
                time.sleep(0.4) # Pequeña pausa dramática para que el video se vea fluido

        # 3. Al terminar el ciclo, guardamos en session y mostramos la rutina final
        st.session_state.final_state = final_state
        
        with final_result_placeholder.container():
            st.markdown("---")
            st.subheader("Rutina Final Entregada")
            st.markdown(final_state.get("final_answer", ""))

    # ==========================================
    # RENDERIZADO ESTÁTICO (Si ya se ejecutó antes)
    # ==========================================
    elif st.session_state.final_state and not submitted:
        state = st.session_state.final_state
        st.success("Ejecución previa cargada. (Ejecute un nuevo prompt para ver la traza en vivo).")
        st.subheader("Rutina Final Entregada")
        st.markdown(state.get("final_answer", ""))

if __name__ == "__main__":
    main()