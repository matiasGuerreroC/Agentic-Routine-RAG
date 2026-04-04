# Agentic-Routine-RAG

Sistema basado en Agentes para la generación de rutinas de entrenamiento físico personalizado utilizando **RAG (Retrieval-Augmented Generation)** y evidencia científica.

## Descripción
Este proyecto resuelve el problema de las rutinas de ejercicio genéricas mediante una arquitectura de agentes inteligentes. El sistema cruza las restricciones del usuario (equipamiento, disponibilidad) con literatura científica de acceso abierto (PubMed, SciELO) para generar planes de entrenamiento seguros y efectivos.

## Stack Tecnológico
- **LLM Principal:** Qwen 3 (32B) vía Groq API.
- **Orquestador:** LangChain.
- **Base de Datos Vectorial:** ChromaDB.
- **APIs Externas:** ExerciseDB & Wger.
- **Backend:** FastAPI (Planificado).

## Hitos del Proyecto
1. **Hito 1:** Recopilación de PDFs y conexión a APIs. (En proceso)
2. **Hito 2:** Implementación del Pipeline RAG.
3. **Hito 3:** Desarrollo del Agente y Tool Calling.
4. **Hito 4:** Evaluación de métricas (Fidelidad y Restricciones).
5. **Hito 5:** Demo final e informe técnico.

## Integrantes
- César Anabalón
- Javier Caamaño
- Matías Guerrero