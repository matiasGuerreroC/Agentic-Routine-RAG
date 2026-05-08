#!/bin/bash
# ===========================================================================
# Script para ejecutar Streamlit App con RAG Avanzado (Unidad 3)
# ===========================================================================
# Este script activa el ambiente virtual y lanza la app Streamlit
# con todos los cambios de RAG 2.0 (MQR + Tríada RAG)

# Verificar que estamos en el directorio correcto
if [ ! -f "src/streamlit_app.py" ]; then
    echo "❌ Error: No se encontró src/streamlit_app.py"
    echo "   Ejecuta este script desde la raíz del proyecto"
    exit 1
fi

# Activar ambiente virtual
echo "🐍 Activando ambiente virtual..."
source venv/Scripts/activate

# Verificar que .env existe
if [ ! -f ".env" ]; then
    echo "⚠️  Advertencia: No se encontró .env"
    echo "   Asegúrate de tener GROQ_API_KEY configurada"
fi

# Verificar que ChromaDB existe
if [ ! -d "chromadb_storage" ]; then
    echo "⚠️  Advertencia: chromadb_storage no existe"
    echo "   Ejecuta primero: python src/ingestion.py"
fi

# Lanzar Streamlit
echo "🚀 Iniciando Streamlit App..."
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "      AGENTIC ROUTINE RAG - UNIDAD 3 (RAG AVANZADO)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✨ Características nuevas:"
echo "   • Multi-Query Retrieval (MQR) automático"
echo "   • Evaluación de Tríada RAG (opcional)"
echo "   • Manejo inteligente de Rate Limits"
echo "   • Interfaz paso a paso mejorada"
echo ""
echo "📖 Instrucciones:"
echo "   1. Escribe tu pregunta (ej: rutina hipertrofia en casa)"
echo "   2. Presiona 'Preparar flujo'"
echo "   3. Sigue los pasos numerados"
echo "   4. En el sidebar: activa 'Evaluar Tríada de RAG' para métricas"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

streamlit run src/streamlit_app.py

# Limpiar virtual environment al cerrar
deactivate
