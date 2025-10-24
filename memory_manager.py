import json
import os
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
from pathlib import Path

from dotenv import load_dotenv

# ---------------- Env ----------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autism_bot")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- Summarization Toggle ----------------
USE_OPENAI = True  # ← Toggle this to False to use local BART model

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------- Summarizer Function ----------------
def summarize_text(text: str) -> str:
    """Summarize text using either OpenAI API or local BART model."""
    try:
        if USE_OPENAI:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        else:
            return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"[Warning] Summarization failed: {e}")
        return text[:400]  # fallback truncation


# ---------------- Memory Manager ----------------
class MemoryManager:
    def __init__(self, user_id: str, base_folder="sessions", max_recent=5):
        self.user_id = user_id
        self.base_folder = Path(base_folder)
        self.base_folder.mkdir(exist_ok=True)
        self.save_path = self.base_folder / f"user_{user_id}.json"
        self.max_recent = max_recent
        self.sessions = {}
        self.load()

    # ----------------------------
    # Load / Save
    # ----------------------------
    def load(self):
        if self.save_path.exists():
            with open(self.save_path, "r", encoding="utf-8") as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2, ensure_ascii=False)
            
    # ----------------------------
    # Create new session
    # ----------------------------
    def create_session(self, user_id: str, user_name: str = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "user_name": user_name,  # ← NEW FIELD
            "created_at": datetime.now().isoformat(),
            "recent": [],
            "packed": [],
        }
        self.save()
        return session_id

    # ----------------------------
    # Add new message
    # ----------------------------
    def add_message(self, session_id: str, role: str, content: str):
        session = self.sessions[session_id]
        session["recent"].append({"role": role, "content": content})

        # If recent exceeds max_recent, pack older messages
        if len(session["recent"]) > self.max_recent:
            self._pack_old_messages(session)

        self.save()

    # ----------------------------
    # Check for recent repetition
    # ----------------------------
    def find_recent_match(self, session_id: str, query: str, threshold=0.85) -> Tuple[bool, str]:
        session = self.sessions[session_id]
        for msg in reversed(session["recent"]):
            if msg["role"] == "user":
                similarity = SequenceMatcher(None, query.lower(), msg["content"].lower()).ratio()
                if similarity >= threshold:
                    idx = session["recent"].index(msg)
                    if idx + 1 < len(session["recent"]) and session["recent"][idx + 1]["role"] == "assistant":
                        return True, session["recent"][idx + 1]["content"]
        return False, None

    # ----------------------------
    # Search packed summaries for related info
    # ----------------------------
    def search_packed(self, session_id: str, query: str) -> Tuple[bool, str]:
        session = self.sessions[session_id]
        best_match = None
        best_score = 0
        for block in session["packed"]:
            sim = SequenceMatcher(None, query.lower(), block["summary"].lower()).ratio()
            if sim > best_score:
                best_score = sim
                best_match = block
        if best_match and best_score > 0.7:
            return True, best_match["summary"]
        return False, None

    # ----------------------------
    # Internal: summarize older messages
    # ----------------------------
    def _pack_old_messages(self, session):
        # Keep last self.max_recent messages; summarize the rest
        if len(session["recent"]) <= self.max_recent:
            return

        to_summarize = session["recent"][:-self.max_recent]
        text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        summary = summarize_text(text)

        session["packed"].append({
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last self.max_recent messages
        session["recent"] = session["recent"][-self.max_recent:]


# ---------------- Example usage ----------------
if __name__ == "__main__":
    mem = MemoryManager(max_recent=11)                                       # CONFIGURABLE
    session_id = mem.create_session("test_user")
    mem.add_message(session_id, "user", "How can I reduce anxiety in my child?")
    mem.add_message(session_id, "assistant", "You can create a calm and predictable environment...")
    print(mem.find_recent_match(session_id, "how can i reduce anxiety in my child"))
