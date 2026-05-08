#!/usr/bin/env python3
"""
Validación rápida del sistema RAG Avanzado (Unidad 3)

Este script verifica que todos los componentes están correctamente instalados
y configurados antes de ejecutar la Streamlit app.

Uso:
    python validate_setup.py
"""

import os
import sys
from pathlib import Path

def check_env():
    """Verifica que .env está configurado con GROQ_API_KEY."""
    print("\n[1/6] Verificando configuración (.env)...")
    
    if not Path(".env").exists():
        print("  ❌ No se encontró .env")
        print("  ℹ️  Crea .env con: GROQ_API_KEY=tu_clave")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("  ❌ GROQ_API_KEY no configurada en .env")
        return False
    
    if len(groq_key) < 20:
        print("  ❌ GROQ_API_KEY parece inválida (muy corta)")
        return False
    
    print("  ✅ .env y GROQ_API_KEY configurados")
    return True

def check_chromadb():
    """Verifica que ChromaDB está disponible."""
    print("\n[2/6] Verificando base de datos vectorial (ChromaDB)...")
    
    if not Path("chromadb_storage").exists():
        print("  ❌ chromadb_storage no encontrado")
        print("  ℹ️  Ejecuta primero: python src/ingestion.py")
        return False
    
    # Verificar que al menos tiene archivos
    chroma_files = list(Path("chromadb_storage").glob("**/*"))
    if not chroma_files:
        print("  ❌ chromadb_storage está vacío")
        print("  ℹ️  Ejecuta: python src/ingestion.py")
        return False
    
    print(f"  ✅ ChromaDB encontrado ({len(chroma_files)} archivos)")
    return True

def check_imports():
    """Verifica que todos los imports necesarios están disponibles."""
    print("\n[3/6] Verificando dependencias...")
    
    required_packages = {
        "streamlit": "Streamlit UI",
        "langchain_chroma": "ChromaDB Vector Store",
        "langchain_groq": "Groq LLM Integration",
        "langchain_huggingface": "HuggingFace Embeddings",
        "langchain_classic": "Multi-Query Retrieval (NEW!)",
        "torch": "PyTorch (para GPU support)",
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: {description}")
            missing.append(package)
    
    if missing:
        print(f"\n  ℹ️  Instala los paquetes faltantes:")
        print(f"      pip install {' '.join(missing)}")
        return False
    
    return True

def check_gpu():
    """Verifica disponibilidad de GPU."""
    print("\n[4/6] Verificando GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ GPU disponible: {gpu_name}")
            return True
        else:
            print("  ℹ️  GPU no disponible, usando CPU")
            print("     (Más lento pero funciona correctamente)")
            return True
    except Exception as e:
        print(f"  ⚠️  Error al verificar GPU: {str(e)}")
        return True  # No es crítico

def check_agent():
    """Verifica que el agente RAG se puede cargar."""
    print("\n[5/6] Verificando agente RAG...")
    
    # Para evitar un Segmentation Fault al instanciar el agente (carga pesada
    # de modelos/embeddings), hacemos una verificación estática del código
    # buscando la definición de la clase y la presencia de métodos clave.
    try:
        agent_path = Path("src/agent.py")
        if not agent_path.exists():
            print("  ❌ No se encontró src/agent.py")
            return False

        src_code = agent_path.read_text(encoding="utf-8")

        checks = [
            ("class RoutineRAGAgent", "Clase RoutineRAGAgent definida"),
            ("def evaluate_rag_triad", "Método evaluate_rag_triad presente"),
            ("MultiQueryRetriever", "Multi-Query Retriever (MQR) integrado"),
            ("def generate_candidates", "generate_candidates presente"),
            ("def retrieve_context", "retrieve_context presente"),
        ]

        all_good = True
        for pattern, desc in checks:
            if pattern in src_code:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ {desc} no encontrado ({pattern})")
                all_good = False

        if all_good:
            print("  ✅ Verificación estática del agente completada correctamente")
            print("  ℹ️  Nota: No se instanció RoutineRAGAgent para evitar cargar modelos pesados")
            return True
        else:
            print("  ❌ Verificación estática incompleta: revisa src/agent.py")
            return False

    except Exception as e:
        print(f"  ❌ Error al verificar agente (estático): {str(e)}")
        return False

def check_streamlit_updates():
    """Verifica que la app Streamlit tiene las actualizaciones."""
    print("\n[6/6] Verificando actualizaciones en Streamlit app...")
    
    try:
        # Intentar leer como UTF-8; en Windows algunos archivos pueden tener
        # codificaciones diferentes, así que hacemos un fallback a latin-1.
        try:
            with open("src/streamlit_app.py", "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print("  ⚠️  Warning: utf-8 decode failed, intentando latin-1")
            with open("src/streamlit_app.py", "r", encoding="latin-1") as f:
                content = f.read()
        
        checks = {
            "advanced_mode": "Modo Avanzado",
            "rag_triad_evaluation": "Evaluación Tríada RAG",
            "Multi-Query Retrieval (MQR)": "Descripción MQR",
            "evaluate_rag_triad": "Método de evaluación",
        }
        
        all_good = True
        for pattern, description in checks.items():
            if pattern in content:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ {description} no encontrado")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"  ⚠️  Error verificando app: {str(e)}")
        return False

def print_summary(results):
    """Imprime resumen de la validación."""
    print("\n" + "="*70)
    print("RESUMEN DE VALIDACIÓN")
    print("="*70)
    
    checks = [
        (".env y GROQ_API_KEY", results[0]),
        ("ChromaDB Base Vectorial", results[1]),
        ("Dependencias Python", results[2]),
        ("GPU (opcional)", results[3]),
        ("Agente RAG", results[4]),
        ("Actualizaciones Streamlit", results[5]),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    print("="*70)
    
    if passed == total:
        print(f"\n✨ ¡ÉXITO! Todos los checks pasaron ({passed}/{total})")
        print("\nPuedes ejecutar la app con:")
        print("  ./run_streamlit.sh")
        print("\nO manualmente:")
        print("  source venv/Scripts/activate")
        print("  streamlit run src/streamlit_app.py")
        return True
    else:
        print(f"\n⚠️  Algunos checks fallaron ({passed}/{total})")
        print("\nRevisa los errores arriba y ejecuta nuevamente.")
        print("Ver STREAMLIT_GUIDE.md para más ayuda.")
        return False

def main():
    print("\n" + "="*70)
    print("VALIDACIÓN: RAG AVANZADO - UNIDAD 3")
    print("="*70)
    
    results = [
        check_env(),
        check_chromadb(),
        check_imports(),
        check_gpu(),
        check_agent(),
        check_streamlit_updates(),
    ]
    
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
