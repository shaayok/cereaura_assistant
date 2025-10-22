import os, re, json, time, hashlib
from pathlib import Path
import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from utils.dialect import detect_leb_chat, normalize_for_embedding
from demo_answers import DEMO_RESPONSES
import numpy as np

# ---------------- Env ----------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autism_bot")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Image Map ----------------
IMG_TOKEN = re.compile(r"\[\[image:([a-zA-Z0-9_\-]+)\]\]")
IMAGE_MAP = {}
IMAGE_MAP_PATH = Path("assets/image_map.json")
if IMAGE_MAP_PATH.exists():
    IMAGE_MAP = json.loads(IMAGE_MAP_PATH.read_text(encoding="utf-8"))

def render_images_from_answer(answer_text: str):
    tags = IMG_TOKEN.findall(answer_text)
    for t in tags:
        if t in IMAGE_MAP:
            st.image(IMAGE_MAP[t], caption=t.replace("_", " ").title(), width=500)

# ---------------- Connect to Chroma ----------------
@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def compose_context(documents, metadatas, cap=12000) -> str:
    out, used = [], 0
    for d, m in zip(documents, metadatas):
        src = m.get("file", m.get("source", "kb"))
        seg = f"\n---\nSource: {src}\n---\n{d}"
        if used + len(seg) > cap:
            break
        out.append(seg)
        used += len(seg)
    return "\n".join(out)

def show_img(path, caption=None, width=500):
    st.image(path, caption=caption, width=width)

collection = get_collection()

# ---------------- Precompute Demo Question Embeddings ----------------
@st.cache_resource
def get_demo_embeddings():
    demo_keys = list(DEMO_RESPONSES.keys())
    embs = client.embeddings.create(
        model="text-embedding-3-small",
        input=demo_keys
    ).data
    emb_vectors = [e.embedding for e in embs]
    return dict(zip(demo_keys, emb_vectors))

DEMO_EMBEDS = get_demo_embeddings()

def find_similar_demo(query: str, threshold: float = 0.88):
    """
    Find the most semantically similar demo question using cosine similarity.
    Returns (best_key, similarity_score)
    """
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = {k: cosine_sim(q_emb, v) for k, v in DEMO_EMBEDS.items()}
    best_key = max(sims, key=sims.get)
    best_score = sims[best_key]
    if best_score >= threshold:
        return best_key, best_score
    return None, None

# ---------------- UI ----------------
st.set_page_config(page_title="Autism Support Assistant", page_icon="🧩", layout="wide")
st.title("🧩 Autism Support Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

chat = st.container()
for m in st.session_state.history:
    with chat.chat_message(m["role"]):
        if m.get("is_demo"):
            # Demo responses
            for block in m["content"]:
                if block["type"] == "text":
                    st.markdown(block["content"], unsafe_allow_html=True)
                elif block["type"] == "image" and block["tag"] in IMAGE_MAP:
                    st.image(
                        IMAGE_MAP[block["tag"]],
                        caption=block["tag"].replace("_", " ").title(),
                        width=500
                    )
        else:
            # Normal GPT or user messages — preserve formatting
            st.markdown(m["content"], unsafe_allow_html=True)

# ---------------- User Input ----------------
user_query = st.chat_input("Type your question here...")
if not user_query:
    st.stop()

normalized_query = user_query.lower().strip()

# ---------------- Semantic Match in Demo Answers ----------------
similar_key, score = find_similar_demo(normalized_query)

# Highlight the "SIMILAR" line for visibility while testing
if similar_key:
    st.markdown(
        f"<div style='padding:8px; border-radius:6px; color:yellow;'>"
        f"<b>SIMILAR:</b> {similar_key}</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<div style='padding:8px; border-radius:6px; color:yellow;'>"
        f"<b>SIMILAR:</b> <i>No Match Found</i></div>",
        unsafe_allow_html=True
    )

if similar_key:
    answer_blocks = DEMO_RESPONSES[similar_key]["answer"]

    with chat.chat_message("user"):
        st.markdown(user_query)
    st.session_state.history.append({"role": "user", "content": user_query})

    s = 4
    x = "Generating response..."
    if similar_key in ["morning routine", "dressing independently"]:
        s = 8
        x = "Generating image..."
    with st.spinner(x):
        time.sleep(s)

    with chat.chat_message("assistant"):
        for block in answer_blocks:
            if block["type"] == "text":
                st.markdown(block["content"], unsafe_allow_html=True)
            elif block["type"] == "image" and block["tag"] in IMAGE_MAP:
                st.image(
                    IMAGE_MAP[block["tag"]],
                    caption=block["tag"].replace("_", " ").title(),
                    width=500
                )

    st.session_state.history.append({
        "role": "assistant",
        "content": answer_blocks,
        "is_demo": True
    })
    st.stop()

# ---------------- Language + Embedding Prep ----------------
is_leb = detect_leb_chat(user_query)
processed_query = normalize_for_embedding(user_query) if is_leb else user_query

with chat.chat_message("user"):
    st.markdown(user_query)
st.session_state.history.append({"role": "user", "content": user_query})

# ---------------- Retrieval ----------------
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query
).data[0].embedding

q = collection.query(query_embeddings=[query_embedding], n_results=12)
docs = q.get("documents", [[]])[0]
metas = q.get("metadatas", [[]])[0]
dists = q.get("distances", [[]])[0] if "distances" in q else [0.0] * len(docs)

triples = [(d, m, s) for d, m, s in zip(docs, metas, dists) if s < 0.4]
if not triples and docs:
    triples = list(zip(docs[:2], metas[:2], dists[:2]))
docs, metas = zip(*[(d, m) for d, m, _ in triples]) if triples else ([], [])

context = compose_context(docs, metas, cap=12000)

# ---------------- Prompt ----------------
system_prompt = """
You are a compassionate autism support guide for parents.
Use KB excerpts if available, otherwise fallback to your knowledge.

ALWAYS reply in this structured HTML format:
<b>Goal:</b> ...
<b>Why it matters:</b> ...
<b>Step-by-step guide:</b> ...
<b>Friendly tip:</b> ...
<b>📚 References & Resources:</b> ...

If visuals are relevant, reference them as [[image:tag]].

End every message with:
"This is guidance, not diagnosis. You can always conduct free screening on our CereAura platform or book a session with a specialized therapist for diagnosis."

Language rules:
- If the parent speaks in Lebanese chat dialect (romanized Arabic like "keef nharak"), reply in the same dialect.
- If the parent writes in English, reply in English.
"""

messages = [{"role": "system", "content": system_prompt}]
for m in st.session_state.history:
    if m.get("is_demo"):
        demo_text = " ".join(
            block["content"] for block in m["content"] if block["type"] == "text"
        )
        messages.append({"role": m["role"], "content": demo_text})
    else:
        messages.append({"role": m["role"], "content": m["content"]})

kb_status = "STRONG_KB" if docs else "NO_KB"
messages.append({
    "role": "user",
    "content": f"Parent's Question: {user_query}\n\n"
               f"Relevant Knowledge Base Excerpts:\n{context if context else 'No relevant excerpts found.'}\n\n"
               f"KB Status: {kb_status}"
})

# ---------------- GPT Response ----------------
with st.spinner("Generating structured response..."):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=2000
    )

answer = resp.choices[0].message.content.strip()

with chat.chat_message("assistant"):
    st.markdown(answer, unsafe_allow_html=True)
    render_images_from_answer(answer)
    if docs:
        with st.expander("📖 Sources", expanded=False):
            for d, m in zip(docs, metas):
                src = m.get("file", m.get("source", "kb"))
                snippet = d[:800] + ("…" if len(d) > 800 else "")
                st.markdown(f"**{src}**\n\n{snippet}")

st.session_state.history.append({"role": "assistant", "content": answer})