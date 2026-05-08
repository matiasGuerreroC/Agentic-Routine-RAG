"""
EJEMPLO: Integración de Evaluación de Tríada RAG en Streamlit

Este archivo muestra cómo usar el nuevo método evaluate_rag_triad()
en la interfaz Streamlit para mostrar métricas de calidad RAG.

Usage:
    1. Ejecuta tu streamlit_app.py actual tal como está
    2. En lugar de ello, puedes descomentar la sección [OPCIÓN AVANZADA]
       para ver las métricas de la Tríada de RAG
"""

import streamlit as st
from src.agent import RoutineRAGAgent

def show_rag_triad_evaluation(agent: RoutineRAGAgent, question: str):
    """
    Muestra un breakdown visual de la evaluación de la Tríada RAG.
    
    Útil para:
    - Debuggear calidad de retrieval
    - Mostrar transparencia del sistema
    - Validar que el contexto es relevante
    """
    
    with st.spinner("Evaluando calidad RAG..."):
        # Ejecutar pipeline completo con evaluación de Tríada
        result = agent.run_pipeline(
            question_spanish=question,
            samples=3,
            include_rag_triad=True
        )
        
        # Extraer resultado
        final_answer = result["final_answer"]
        evaluation = result.get("rag_triad_evaluation", {})
        
        # Mostrar respuesta
        st.markdown("## 🎯 Rutina Recomendada")
        st.markdown(final_answer)
        
        # ========== SECCIÓN OPCIONAL: MÉTRICAS RAG ==========
        if evaluation and "score_general" in evaluation:
            st.markdown("---")
            st.markdown("## 📊 Evaluación de Calidad RAG (Tríada)")
            
            col1, col2, col3 = st.columns(3)
            
            # Métrica 1: Relevancia del Contexto
            with col1:
                contexto_score = evaluation.get("relevancia_contexto", 0)
                st.metric(
                    "Relevancia Contexto",
                    f"{contexto_score}/5",
                    delta=contexto_score - 3,  # Comparar con neutral (3)
                )
                st.caption(evaluation.get("justificacion_contexto", ""))
            
            # Métrica 2: Fidelidad
            with col2:
                fidelidad_score = evaluation.get("fidelidad", 0)
                st.metric(
                    "Fidelidad",
                    f"{fidelidad_score}/5",
                    delta=fidelidad_score - 3,
                )
                st.caption(evaluation.get("justificacion_fidelidad", ""))
            
            # Métrica 3: Relevancia de Respuesta
            with col3:
                respuesta_score = evaluation.get("relevancia_respuesta", 0)
                st.metric(
                    "Relevancia Respuesta",
                    f"{respuesta_score}/5",
                    delta=respuesta_score - 3,
                )
                st.caption(evaluation.get("justificacion_respuesta", ""))
            
            # Score General
            st.markdown("---")
            score_general = evaluation.get("score_general", 0)
            col_score = st.columns([1, 3])[0]
            with col_score:
                st.metric("Score General", f"{score_general}/5")
            
            # Recomendaciones
            if "recomendaciones" in evaluation:
                st.info(f"💡 **Observaciones:** {evaluation['recomendaciones']}")


# ========== EJEMPLO DE USO EN STREAMLIT ==========
if __name__ == "__main__":
    st.set_page_config(page_title="RAG Avanzado con Evaluación", layout="wide")
    st.title("🏋️ Generador de Rutinas de Entrenamiento (RAG 2.0)")
    st.markdown("Sistema con Multi-Query Retrieval y Evaluación de Tríada RAG")
    
    # Inicializar agent
    agent = RoutineRAGAgent(k=4)
    
    # Input del usuario
    question = st.text_area(
        "¿Cuál es tu pregunta sobre entrenamiento?",
        placeholder="Ej: Rutina de hipertrofia en casa sin pesas por 3 días a la semana",
        height=100
    )
    
    # Botón para generar
    if st.button("🚀 Generar Rutina con Evaluación RAG"):
        if question.strip():
            show_rag_triad_evaluation(agent, question)
        else:
            st.warning("Por favor, ingresa una pregunta.")
    
    # Sidebar con info
    with st.sidebar:
        st.markdown("### ℹ️ Sobre este Sistema")
        st.markdown("""
        **Arquitectura RAG Avanzada:**
        - ✅ Multi-Query Retrieval (MQR)
        - ✅ Embeddings Semánticos (Nomic)
        - ✅ Self-Consistency (3 candidatos)
        - ✅ Evaluación de Tríada RAG
        
        **Métricas Mostradas:**
        1. Relevancia Contexto: ¿Los docs recuperados son útiles?
        2. Fidelidad: ¿La respuesta cita el contexto?
        3. Relevancia Respuesta: ¿Contesta la pregunta?
        """)
