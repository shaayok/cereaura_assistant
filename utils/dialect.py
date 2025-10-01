LEB_PATTERNS = {
    "keef": "how",
    "kifak": "how are you",
    "nharak": "your day",
    "lyom": "today",
    "mnee7": "good",
    "shou": "what",
    "akhbarak": "your news",
    "habib": "dear",
    "sah": "right"
}

def normalize_for_embedding(text: str) -> str:
    """
    Replace Lebanese romanized tokens with English so embeddings work better.
    """
    out = text
    for pat, repl in LEB_PATTERNS.items():
        out = out.replace(pat, repl).replace(pat.capitalize(), repl)
    return out

def detect_leb_chat(text: str) -> bool:
    low = text.lower()
    return any(tok in low for tok in LEB_PATTERNS.keys())
