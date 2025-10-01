import os, re, json
from pathlib import Path
import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from utils.dialect import detect_leb_chat, normalize_for_embedding
from demo_answers import DEMO_RESPONSES
import time


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
            st.image(IMAGE_MAP[t], caption=t.replace("_"," ").title(), width=500)

# ---------------- Connect to Chroma ----------------
@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def compose_context(documents, metadatas, cap=12000) -> str:
    """
    Format KB chunks into a single context string with source labels,
    capped at a total character length.
    """
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
    # width controls size; do NOT pass use_container_width if you want fixed pixel width
    st.image(path, caption=caption, width=width)



collection = get_collection()

# ---------------- UI ----------------
st.set_page_config(page_title="Autism Support Assistant", page_icon="🧩", layout="wide")
st.title("🧩 Autism Support Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

chat = st.container()
for m in st.session_state.history:
    with chat.chat_message(m["role"]):
        if m.get("is_demo"):
            # Replay demo answers with block rendering
            for block in m["content"]:
                if block["type"] == "text":
                    st.markdown(block["content"], unsafe_allow_html=True)
                elif block["type"] == "image" and block["tag"] in IMAGE_MAP:
                    st.image(
                        IMAGE_MAP[block["tag"]],
                        caption=block["tag"].replace("_"," ").title(),
                        width=500
                    )
        else:
            # Normal GPT or user messages
            st.markdown(m["content"])


user_query = st.chat_input("Type your question here...")
if not user_query:
    st.stop()
normalized_query = user_query.lower().strip()
# Exact or partial match
for key, resp_data in DEMO_RESPONSES.items():
    if key in normalized_query:
        answer_blocks = resp_data["answer"]

        # Show user bubble first
        with chat.chat_message("user"):
            st.markdown(user_query)
        st.session_state.history.append({"role": "user", "content": user_query})
        s = 4
        x = "Generating response..."
        if key  == "morning routine" or key == "dressing independently":
            s = 8
            x = "Generating image..."
        with st.spinner(x):
            time.sleep(s)


        # Then show assistant bubble
        with chat.chat_message("assistant"):
            # Loop through blocks: text and images
            for block in answer_blocks:
                if block["type"] == "text":
                    st.markdown(block["content"], unsafe_allow_html=True)
                elif block["type"] == "image":
                    if block["tag"] in IMAGE_MAP:
                        st.image(
                            IMAGE_MAP[block["tag"]],
                            caption=block["tag"].replace("_"," ").title(),
                            width=500
                        )

        # Save full content as a string (for history)
        joined_text = " ".join(
            block["content"] for block in answer_blocks if block["type"] == "text"
        )
        st.session_state.history.append({
            "role": "assistant",
            "content": answer_blocks,   # keep full block list
            "is_demo": True
        })


        st.stop()  # Skip GPT, return immediately


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

# filter by distance (lower is better)
# Soften the threshold: allow more chunks in
triples = [(d, m, s) for d, m, s in zip(docs, metas, dists) if s < 0.4]

# If nothing passes the filter, still keep the top 2 raw chunks as a fallback context
if not triples and docs:
    triples = list(zip(docs[:2], metas[:2], dists[:2]))

docs, metas = zip(*[(d, m) for d, m, _ in triples]) if triples else ([], [])


# build context string (see updated compose_context below)
context = compose_context(docs, metas, cap=12000)

# ---------------- Prompt ----------------
system_prompt = """
You are a compassionate autism support guide for parents.
- Use KB excerpts if available, otherwise fallback to your knowledge.
- If visuals are relevant, reference them as [[image:tag]].
- Write detailed, clear, and supportive answers.
- Always add in the end: "This is guidance, not diagnosis. You can always conduct free screening on our CereAura platform or book a session with a specialized therapist for diagnosis."

Language rules:
- If the parent speaks in Lebanese chat dialect (romanized Arabic like "keef nharak"), reply in the same Lebanese chat dialect.
- Do not switch to Egyptian or other dialects.
- If the parent writes in English, reply in English.
"""

messages = [{"role": "system", "content": system_prompt}]
for m in st.session_state.history:
    if m.get("is_demo"):
        # Demo answers: flatten to text so GPT can use them as context
        demo_text = " ".join(
            block["content"] for block in m["content"] if block["type"] == "text"
        )
        messages.append({"role": m["role"], "content": demo_text})
    else:
        # Normal user or GPT answers
        messages.append({"role": m["role"], "content": m["content"]})
kb_status = "STRONG_KB" if docs else "NO_KB"
messages.append({
    "role": "user",
    "content": f"Parent's Question: {user_query}\n\n"
               f"Relevant Knowledge Base Excerpts:\n{context if context else 'No relevant excerpts found.'}\n\n"
               f"KB Status: {kb_status}"
})



# Spinner while waiting for GPT
with st.spinner("Generating response..."):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=2000
    )

answer = resp.choices[0].message.content.strip()

with chat.chat_message("assistant"):
    st.markdown(answer)
    render_images_from_answer(answer)
    if docs:
        with st.expander("📖 Sources", expanded=False):
            for d, m in zip(docs, metas):
                src = m.get("file", m.get("source", "kb"))
                snippet = d[:800] + ("…" if len(d) > 800 else "")
                st.markdown(f"**{src}**\n\n{snippet}")

st.session_state.history.append({"role": "assistant", "content": answer})
