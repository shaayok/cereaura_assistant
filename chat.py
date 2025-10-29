import os, re, json, time, hashlib
from pathlib import Path
import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from utils.dialect import detect_leb_chat, normalize_for_embedding
from demo_answers import DEMO_RESPONSES
import numpy as np
from memory_manager import MemoryManager

# ---------------- User Login ----------------
from user_login import login_page

user_data = login_page()
if not user_data:
    st.stop()                           # wait until user logs in or registers

# Once logged in:
user_id = user_data["id"]
user_name = user_data["name"]
has_autistic_child = user_data["has_autistic_child"]

# ---------------- LOGOUT FUNCTIONALITY ----------------
def logout():
    """Clear session state and force a rerun to go back to login."""
    # Clear the specific login/user data keys
    for key in ["user_data", "user_id", "session_id", "history", "greeted", "memory_manager"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- CSS to style the button and make it sticky top-right ---
st.markdown(
    """
    <style>
    /* 1. Target the div containing the button and make it sticky (fixed position) */
    div.row-widget.stButton:has(> button#logout_button_id) {
        position: fixed;
        top: 10px;
        right: 20px; /* Adjusted from 10px for better margin */
        z-index: 1000;
    }
    
    /* 2. Style the actual Streamlit button */
    .stButton>button#logout_button_id {
        background-color: #FF4B4B; /* Red background color */
        color: white; /* White text color */
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
    }

    /* 3. Style the button on hover */
    .stButton>button#logout_button_id:hover {
        background-color: #e04444; /* Slightly darker red on hover */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Place the Streamlit button with a specific 'key' that matches the CSS ID
if st.button("🔴 Log Out", key="logout_button_id"):
    logout()
# ---------------- END LOGOUT FUNCTIONALITY ----------------


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
    """Find the most semantically similar demo question using cosine similarity."""
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

# ---------------- Memory Manager (Per-User Session Files) ----------------
if "memory_manager" not in st.session_state or st.session_state.get("user_id") != user_id:
    st.session_state.memory_manager = MemoryManager(user_id)  # ← pass user_id to manage their own session file
    st.session_state.user_id = user_id
    st.session_state.session_id = st.session_state.memory_manager.create_session(user_id, user_name)
    st.session_state.history = []

# ---------------- First-time Personalized Greeting ----------------
if "greeted" not in st.session_state or not st.session_state.greeted:
    if has_autistic_child == 0:
        greeting = f"""
<b>Goal:</b> Welcome {user_name}! I’m your Autism Support Assistant. I'm here to help you support your child's unique journey.  
<br><b>Why it matters:</b> Parenting a child on the spectrum comes with challenges, but also countless moments of joy and growth. Having the right support makes a big difference.  
<br><b>Step-by-step guide:</b>  
1. You can ask me questions about communication, behavior, therapy, or parenting strategies.  
2. I’ll provide structured, compassionate guidance every time.  
3. You can revisit past chats — I remember what we discussed. 

<br><b>Friendly tip:</b> How is your child doing today? 😊  
<br><b>📚 References & Resources:</b> Autism Parenting Magazine, CDC Developmental Milestones, CereAura Platform.  
<br><i>This is guidance, not diagnosis. You can always conduct free screening on our CereAura platform or book a session with a specialized therapist for diagnosis.</i>
"""
    else:
        greeting = f"""
<b>Goal:</b> Hello {user_name}! I’m your Autism Support Assistant — here to guide you in understanding autism and supporting families affected by it.  
<br><b>Why it matters:</b> Building awareness and empathy helps create inclusive, supportive environments for individuals on the spectrum.  
<br><b>Step-by-step guide:</b>  
1. You can ask me about early signs, interventions, or how to support others.  
2. I’ll provide well-structured, evidence-based information.  
3. You can revisit past chats — I remember what we discussed.  

<br><b>Friendly tip:</b> What kind of guidance are you looking for today? 🌟  
<br><b>📚 References & Resources:</b> Autism Speaks, WHO Resources, CereAura Platform.  
<br><i>This is guidance, not diagnosis. You can always conduct free screening on our CereAura platform or book a session with a specialized therapist for diagnosis.</i>
"""

    # Add greeting to memory/history, but do NOT display it yet
    st.session_state.history = [{"role": "assistant", "content": greeting}]
    st.session_state.memory_manager.add_message(st.session_state.session_id, "assistant", greeting)
    st.session_state.greeted = True
    # st.stop()

chat = st.container()
for m in st.session_state.history:
    with chat.chat_message(m["role"]):
        if m.get("is_demo"):
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
            st.markdown(m["content"], unsafe_allow_html=True)

# ---------------- User Input ----------------
user_query = st.chat_input("Type your question here...")
if not user_query:
    st.stop()

normalized_query = user_query.lower().strip()

# ---------------- Memory: Check for Repeat Query ----------------
found, cached_answer = st.session_state.memory_manager.find_recent_match(
    st.session_state.session_id, user_query
)
if found:
    with chat.chat_message("user"):
        st.markdown(user_query)
    st.session_state.history.append({"role": "user", "content": user_query})

    with chat.chat_message("assistant"):
        st.markdown(cached_answer, unsafe_allow_html=True)

    st.session_state.history.append({"role": "assistant", "content": cached_answer})
    st.stop()

# ---------------- Semantic Match in Demo Answers ----------------
similar_key, score = find_similar_demo(normalized_query)

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

# ---------------- Memory: Search Packed Summaries ----------------
found_summary, related_info = st.session_state.memory_manager.search_packed(
    st.session_state.session_id, user_query
)
if found_summary:
    st.markdown(
        f"<div style='padding:8px; border-radius:6px; color:lightgreen;'>"
        f"<b>Found related past topic in memory:</b> {related_info}</div>",
        unsafe_allow_html=True
    )

# ---------------- DEMO ANSWER HANDLER ----------------
if similar_key:
    answer_blocks = DEMO_RESPONSES[similar_key]["answer"]

    with chat.chat_message("user"):
        st.markdown(user_query)
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.memory_manager.add_message(
        st.session_state.session_id, "user", user_query
    )

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

    # Store demo response in memory
    text_answer = " ".join(
        block["content"] for block in answer_blocks if block["type"] == "text"
    )
    st.session_state.memory_manager.add_message(
        st.session_state.session_id, "assistant", text_answer
    )

    st.stop()

# ---------------- Language + Embedding Prep ----------------
is_leb = detect_leb_chat(user_query)
processed_query = normalize_for_embedding(user_query) if is_leb else user_query

with chat.chat_message("user"):
    st.markdown(user_query)
st.session_state.history.append({"role": "user", "content": user_query})
st.session_state.memory_manager.add_message(
    st.session_state.session_id, "user", user_query
)

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

if found_summary:
    messages.append({
        "role": "system",
        "content": f"Here is a past related summary from previous interactions:\n{related_info}"
    })

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
st.session_state.memory_manager.add_message(
    st.session_state.session_id, "assistant", answer
)