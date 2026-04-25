from langdetect import detect
import pathlib

def audit_languages():
    print("--- Auditoría de Idiomas ---")
    
    contador = {}
    
    for md_path in pathlib.Path("data/markdowns/").glob("*.md"):
        content = md_path.read_text(encoding="utf-8")[:2000]
        lang = detect(content)
        print(f"File: {md_path.name[:30]}... | Detected: {lang}")
        contador[lang] = contador.get(lang, 0) + 1

    print("\n--- Conteo de Idiomas ---")
    for lang, count in contador.items():
        print(f"{lang}: {count}")

audit_languages()