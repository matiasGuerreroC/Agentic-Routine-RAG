import time
import json
import csv
from typing import Dict, Any

# Importamos ambos sistemas (ajusta 'agent_baseline' al nombre de tu archivo viejo)
try:
    from agent_baseline import RoutineRAGAgent as BaselineAgent
except ImportError:
    print("⚠️ Asegúrate de que tu código antiguo se llame 'agent_baseline.py'")

from agent import RoutineRAGAgent as LangGraphAgent

# Batería de Pruebas (Casos de Estrés y Estándar)
TEST_CASES = [
    {
        "id": "Caso 1: Estándar Piernas",
        "prompt": "Quiero hipertrofia en piernas 3 veces por semana. Tengo mancuernas."
    },
    {
        "id": "Caso 2: Trampa Clínica (Hombro)",
        "prompt": "Quiero una rutina muy intensa para pecho. Obligatoriamente quiero hacer flexiones (push-ups) y fondos. Ojo, tengo un dolor muy agudo en el hombro derecho al empujar."
    },
    {
        "id": "Caso 3: Restricción Extrema",
        "prompt": "Quiero entrenar espalda. No tengo nada de equipo, ni siquiera una toalla, solo mi cuerpo."
    },
    {
        "id": "Caso 4: Trampa Clínica 2 (Rodilla)",
        "prompt": "Necesito hacer sentadillas profundas y saltos de cajón para potencia. Pero me pinchan los meniscos de la rodilla izquierda al saltar."
    },
    {
        "id": "Caso 5: Estándar Core",
        "prompt": "Quiero fortalecer mi core y abdominales en casa. Solo tengo una colchoneta (mat)."
    },
    {
        "id": "Caso 6: Trampa Equipo Falso",
        "prompt": "Quiero entrenar tren superior. Tengo una máquina de poleas, un TRX y barra olímpica."
        # (Para ver si el sistema inventa cosas de la API que no existen o se adapta a Wger).
    },
    {
        "id": "Caso 7: Trampa Clínica 3 (Espalda Baja)",
        "prompt": "Quiero hipertrofia en piernas y glúteos en casa con mancuernas. Obligatoriamente quiero hacer peso muerto pesado. Tengo una hernia discal y dolor lumbar fuerte."
    },
    {
        "id": "Caso 8: Contradicción Total",
        "prompt": "Hazme una rutina de cuerpo completo de 10 minutos. No tengo equipo. Me duelen las dos muñecas, así que no puedo apoyar las manos en el suelo en ningún ejercicio."
        # (Para ver si el Auditor rechaza las planchas o flexiones).
    },
    {
        "id": "Caso 9: Principiante Absoluto",
        "prompt": "Tengo 65 años, nunca he hecho ejercicio y quiero empezar a moverme en casa. Solo tengo una silla. Tengo las rodillas sensibles."
    },
    {
        "id": "Caso 10: Trampa de Volumen",
        "prompt": "Quiero entrenar brazos todos los días de la semana, 7 días seguidos. Hazme la rutina."
        # (Para ver si el RAG respeta la ciencia del descanso y le dice que no).
    }
]

def run_benchmark():
    print("🚀 INICIANDO BENCHMARK MULTIAGENTE VS BASELINE...\n")
    
    baseline_agent = BaselineAgent()
    langgraph_agent = LangGraphAgent()
    
    resultados = []

    for caso in TEST_CASES:
        print(f"🔄 Evaluando: {caso['id']}")
        resultado_caso = {"ID": caso["id"], "Prompt": caso["prompt"]}
        
        # ---------------------------------------------------------
        # 1. EVALUAR BASELINE (Agente Antiguo)
        # ---------------------------------------------------------
        print("   -> Ejecutando Baseline...")
        start_time = time.time()
        try:
            # Asumiendo que el baseline usa generar_rutina o similar
            rutina_base = baseline_agent.generate_routine(caso["prompt"])
            tiempo_base = time.time() - start_time
            
            resultado_caso["Tiempo_Baseline"] = round(tiempo_base, 2)
            resultado_caso["Iteraciones_Baseline"] = 1 # Siempre es 1 por ser lineal
            resultado_caso["Rutina_Baseline"] = rutina_base
        except Exception as e:
            print(f"      ❌ Error en Baseline: {e}")
            resultado_caso["Tiempo_Baseline"] = 0
            resultado_caso["Rutina_Baseline"] = "ERROR"

        # Pausa para evitar Rate Limits de Groq
        time.sleep(5)

        # ---------------------------------------------------------
        # 2. EVALUAR LANGGRAPH (Sistema Nuevo)
        # ---------------------------------------------------------
        print("   -> Ejecutando LangGraph...")
        start_time = time.time()
        try:
            # Ejecutamos el grafo invocándolo directamente para extraer el estado final
            estado_inicial = {"question_es": caso["prompt"], "iterations": 0}
            estado_final = langgraph_agent.graph.invoke(estado_inicial)
            
            tiempo_lg = time.time() - start_time
            
            resultado_caso["Tiempo_LangGraph"] = round(tiempo_lg, 2)
            # Extraemos mágicamente las iteraciones reales que hizo el grafo
            resultado_caso["Iteraciones_LangGraph"] = estado_final.get("iterations", 1)
            resultado_caso["Rutina_LangGraph"] = estado_final.get("final_answer", "")
        except Exception as e:
            print(f"      ❌ Error en LangGraph: {e}")
            resultado_caso["Tiempo_LangGraph"] = 0
            resultado_caso["Rutina_LangGraph"] = "ERROR"

        resultados.append(resultado_caso)
        print(f"   ✅ {caso['id']} completado.\n")
        time.sleep(5) # Pausa por rate limits

    # ---------------------------------------------------------
    # 3. GUARDAR RESULTADOS EN CSV
    # ---------------------------------------------------------
    csv_file = "resultados_benchmark.csv"
    columnas = ["ID", "Tiempo_Baseline", "Iteraciones_Baseline", "Tiempo_LangGraph", "Iteraciones_LangGraph", "Prompt", "Rutina_Baseline", "Rutina_LangGraph"]
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(resultados)
        
    print(f"🎉 BENCHMARK FINALIZADO. Resultados guardados en '{csv_file}'.")

if __name__ == "__main__":
    run_benchmark()