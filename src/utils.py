import re

def clean_llm_response(text: str) -> str:
    # Elimina todo lo que esté entre los tags <think> y </think>
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return clean_text.strip()