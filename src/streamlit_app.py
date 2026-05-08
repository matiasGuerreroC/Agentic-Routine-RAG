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
        max-width: 1180px;
    }
    .step-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #5f6b7a;
        font-size: 0.95rem;
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
        "question_es": "",
        "question_en": "",
        "context": "",
        "candidates": [],
        "final_answer": "",
        "stage": 0,
        "samples": 3,
        "rag_triad_evaluation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline(question: str, samples: int) -> None:
    st.session_state.question_es = question.strip()
    st.session_state.question_en = ""
    st.session_state.context = ""
    st.session_state.candidates = []
    st.session_state.final_answer = ""
    st.session_state.stage = 0
    st.session_state.samples = samples
    st.session_state.rag_triad_evaluation = None


def render_step_header(number: int, title: str, done: bool = False) -> None:
    icon = "✅" if done else f"{number}."
    st.markdown(
        f'<div class="step-card"><strong>{icon} {title}</strong></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    agent = get_agent()

    st.title("Agentic Routine RAG")
    st.write(
        "Interfaz paso a paso para traducir la consulta, recuperar evidencia, generar candidatos y elegir la rutina final."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "No se detectó GROQ_API_KEY en el entorno. El agente no podrá generar respuestas hasta configurarla."
        )

    with st.sidebar:
        st.header("Configuración")
        st.write("Modelo LLM: qwen/qwen3-32b")
        st.write("Embedding: nomic-ai/nomic-embed-text-v1.5")
        st.write(f"Base vectorial: {CHROMA_PATH}")
        samples = 3
        st.markdown("**Cantidad de candidatos:** 3")
        st.caption("Se usa una evaluación final con self-consistency sobre tres candidatos.")
        
        st.markdown("---")
        st.info(
            "📊 **RAG Avanzado Automático:**\n"
            "Se evaluarán 3 métricas de calidad (Tríada RAG) automáticamente al finalizar: "
            "Relevancia Contexto, Fidelidad y Relevancia Respuesta."
        )

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Pregunta del usuario en español",
            height=180,
            value=st.session_state.question_input,
            placeholder="Describe tu objetivo, equipo disponible, días de entrenamiento y molestias si las hay.",
        )
        submitted = st.form_submit_button("Preparar flujo", use_container_width=True)

    if submitted:
        st.session_state.question_input = question
        reset_pipeline(question, samples)
        st.rerun()

    if not st.session_state.question_es:
        st.info("Escribe una pregunta y presiona 'Preparar flujo' para comenzar.")
        return

    st.progress(st.session_state.stage / 4)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Entrada")
        st.markdown(f"**Pregunta en español:**\n\n{st.session_state.question_es}")
    with cols[1]:
        st.subheader("Estado")
        st.markdown(
            f"- Paso actual: **{st.session_state.stage} / 4**\n"
            f"- Candidatos por evaluar: **{st.session_state.samples}**\n"
            f"- GPU disponible: **{'sí' if torch.cuda.is_available() else 'no'}**"
        )

    st.markdown("---")

    # Paso 0: traducción
    render_step_header(0, "Traducción ES -> EN", done=st.session_state.stage > 0)
    if st.session_state.stage == 0:
        st.caption("Traduce la consulta al inglés para que la búsqueda semántica encaje mejor con el embedding.")
        if st.button("Traducir y continuar", use_container_width=True):
            with st.spinner("Traduciendo pregunta..."):
                st.session_state.question_en = agent.translate_question(st.session_state.question_es)
            st.session_state.stage = 1
            st.rerun()
    else:
        st.text_area("Traducción al inglés", value=st.session_state.question_en, height=120, disabled=True)

    # Paso 1: retrieval
    render_step_header(1, "Recuperación RAG", done=st.session_state.stage > 1)
    if st.session_state.stage >= 1:
        if st.session_state.stage == 1:
            st.caption(
                "🔍 Recupera contexto científico con Multi-Query Retrieval (MQR): "
                "genera 3 variantes técnicas en inglés antes de buscar."
            )
            if st.button("Recuperar contexto y continuar", use_container_width=True):
                with st.spinner("Recuperando evidencia..."):
                    st.session_state.context = agent.retrieve_context(st.session_state.question_en)
                st.session_state.stage = 2
                st.rerun()
        else:
            with st.expander("Ver contexto recuperado", expanded=False):
                st.text_area("Contexto científico", value=st.session_state.context, height=320, disabled=True)

    # Paso 2: generación
    render_step_header(2, "Generación de candidatos", done=st.session_state.stage > 2)
    if st.session_state.stage >= 2:
        if st.session_state.stage == 2:
            st.caption("Genera varias respuestas para escoger la más segura y consistente.")
            if st.button("Generar candidatos y continuar", use_container_width=True):
                with st.spinner("Generando candidatos..."):
                    st.session_state.candidates = agent.generate_candidates(
                        question_spanish=st.session_state.question_es,
                        question_english=st.session_state.question_en,
                        context=st.session_state.context,
                        samples=st.session_state.samples,
                    )
                st.session_state.stage = 3
                st.rerun()
        else:
            for index, candidate in enumerate(st.session_state.candidates, start=1):
                with st.expander(f"Candidato {index}", expanded=index == 1):
                    st.markdown(candidate)

    # Paso 3: evaluación final
    render_step_header(3, "Evaluación final", done=st.session_state.stage > 3)
    if st.session_state.stage >= 3:
        if st.session_state.stage == 3:
            st.caption("El juez selecciona la mejor opción de los candidatos generados. Luego se evalúa automáticamente la Tríada RAG.")
            if st.button("Evaluar y finalizar", use_container_width=True):
                with st.spinner("Evaluando candidatos y Tríada RAG..."):
                    st.session_state.final_answer = agent.judge_candidates(
                        st.session_state.question_es,
                        st.session_state.candidates,
                    )

                    # Evaluación de Tríada RAG siempre se ejecuta
                    try:
                        docs_list = []
                        if hasattr(agent._retriever, 'invoke'):
                            docs_list = agent._retriever.invoke(st.session_state.question_en)

                        if docs_list:
                            st.session_state.rag_triad_evaluation = agent.evaluate_rag_triad(
                                question_spanish=st.session_state.question_es,
                                answer=st.session_state.final_answer,
                                docs_retrieved=docs_list,
                            )
                        else:
                            # Si no hay docs, la evaluación heurística se ejecuta igual
                            st.session_state.rag_triad_evaluation = agent.evaluate_rag_triad(
                                question_spanish=st.session_state.question_es,
                                answer=st.session_state.final_answer,
                                docs_retrieved=[],
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Error en evaluación de Tríada: {str(e)}")

                st.session_state.stage = 4
                st.rerun()
        else:
            st.success("Rutina final lista.")
            st.markdown(st.session_state.final_answer)

            # Mostrar métricas de Tríada RAG (siempre disponible)
            if st.session_state.rag_triad_evaluation:
                st.markdown("---")
                st.subheader("📊 Evaluación de Calidad RAG (Tríada)")

                eval_data = st.session_state.rag_triad_evaluation

                # Tres columnas con métricas principales
                col1, col2, col3 = st.columns(3)

                with col1:
                    rel_contexto = eval_data.get("relevancia_contexto", 0)
                    st.metric(
                        "Relevancia Contexto",
                        f"{rel_contexto}/5",
                        delta=rel_contexto - 3,
                    )
                    st.caption(eval_data.get("justificacion_contexto", ""))

                with col2:
                    fidelidad = eval_data.get("fidelidad", 0)
                    st.metric(
                        "Fidelidad",
                        f"{fidelidad}/5",
                        delta=fidelidad - 3,
                    )
                    st.caption(eval_data.get("justificacion_fidelidad", ""))

                with col3:
                    rel_respuesta = eval_data.get("relevancia_respuesta", 0)
                    st.metric(
                        "Relevancia Respuesta",
                        f"{rel_respuesta}/5",
                        delta=rel_respuesta - 3,
                    )
                    st.caption(eval_data.get("justificacion_respuesta", ""))

                # Score general
                st.markdown("---")
                score_general = eval_data.get("score_general", 0)
                st.metric("Score General RAG", f"{score_general}/5")

                # Recomendaciones
                if "recomendaciones" in eval_data and eval_data["recomendaciones"]:
                    st.info(f"💡 **Observaciones:** {eval_data['recomendaciones']}")

            st.download_button(
                "Descargar respuesta en Markdown",
                data=st.session_state.final_answer,
                file_name="rutina_final.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if st.session_state.stage == 4:
        st.markdown("---")
        st.success("Proceso completado. Puedes editar la pregunta y volver a preparar el flujo cuando quieras.")


if __name__ == "__main__":
    main()
