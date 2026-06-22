# Agentic-Routine-RAG

Sistema Multiagente para la generación de rutinas de entrenamiento físico personalizado utilizando **RAG (Retrieval-Augmented Generation)**, APIs externas y **LangGraph**.

## Descripción del Proyecto

Este proyecto resuelve el problema de las rutinas de ejercicio genéricas y el riesgo clínico asociado al **"Sesgo de Complacencia"** de los LLMs. Cuando un usuario exige rutinas lesivas, los modelos tradicionales tienden a obedecer. Para mitigar este riesgo, implementamos una arquitectura de **Agentes Inteligentes con Patrón de Reflexión (Auto-corrección)**.

El sistema cruza las restricciones del usuario (dolores, equipamiento) con literatura científica (49 papers en inglés) y un catálogo real de ejercicios para generar planes de entrenamiento. Si la rutina propuesta representa un riesgo para el usuario, un **Auditor Médico** independiente bloquea la entrega y obliga al sistema a reescribirla.

## Arquitectura del Sistema (LangGraph)

El flujo está orquestado mediante un `StateGraph` que maneja una memoria transaccional (`AgentState`). El sistema se aísla cognitivamente en dos tipos de nodos:

**1. Nodos Cognitivos (Cerebros LLM):**
*   **Analizador (Qwen 3 32B):** Traduce la consulta al inglés técnico y perfila al usuario (detectando lesiones o dolores).
*   **Generador (Qwen 3 32B):** Agente creativo que cruza la evidencia y el catálogo para redactar la rutina.
*   **Auditor Clínico (Llama 3.3 70B):** Juez estricto. Evalúa el riesgo biomecánico de la rutina. Tiene poder de veto absoluto.

**2. Nodos Deterministas (Herramientas):**
*   **Tool RAG (ChromaDB + Nomic):** Implementa *Multi-Query Retriever (MQR)* para expandir la búsqueda semántica.
*   **Tool API (Wger):** Extrae ejercicios mediante mapeo dinámico. Cuenta con un sistema de *Fallback* a un diccionario local para garantizar resiliencia ante caídas de red (Error 500 / Timeout).
*   **Formateador:** Limpia el texto mediante Expresiones Regulares (Regex) e inyecta advertencias médicas ineludibles.

## Stack Tecnológico

- **Orquestador:** LangGraph & LangChain.
- **Modelos de Inferencia (vía Groq API):** `qwen/qwen3-32b` (Generación) y `llama-3.3-70b-versatile` (Auditoría).
- **Embeddings:** `nomic-ai/nomic-embed-text-v1.5`.
- **Base de Datos Vectorial:** ChromaDB.
- **APIs Externas:** Wger REST API.
- **Frontend / UI:** Streamlit (Renderizado asíncrono y efecto cascada de nodos).

## Instalación y Ejecución

1. Clonar el repositorio e instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Crear un archivo `.env` en la raíz del proyecto y agregar tu API Key de Groq:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
3. **Frontend Interactivo (Recomendado):** Muestra la ejecución del grafo y mutación de estado en tiempo real.
   ```bash
   streamlit run src/streamlit_app.py
   ```
4. **Backend por consola:**
   ```bash
   python src/agent.py
   ```
5. **Ejecución del Benchmark (A/B Testing):** Compara el rendimiento y seguridad del modelo LangGraph vs un Agente Secuencial Baseline.
   ```bash
   python src/benchmark.py
   ```

## Validación Experimental (Benchmark)

El sistema fue sometido a pruebas de estrés clínico ("Trampas Médicas"). Demostró ser superior a arquitecturas de un solo disparo (Zero-Shot), elevando la **Tasa de Seguridad Clínica del 60% al 100%**, logrando bloquear y corregir ejercicios lesivos (ej. prescribir sentadillas con dolor de meniscos). Además, el sistema iterativo demostró ser **2.4x más rápido** que enfoques de fuerza bruta (*Self-Consistency*).

## Integrantes

- César Anabalón
- Javier Caamaño
- Matías Guerrero

*Proyecto desarrollado para la asignatura LLMs: Fundamentos y Práctica - PUCV (Primer Semestre 2026).*
