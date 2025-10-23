import streamlit as st
import chromadb
from pathlib import Path

# ---------------- CONFIG ----------------
CHROMA_PATH = Path("db")
USER_COLLECTION = "users"
START_ID = 1000
MAX_ID = 9999


# ---------------- INIT CHROMA ----------------
@st.cache_resource
def get_user_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=USER_COLLECTION)


# ---------------- HELPERS ----------------
def generate_user_id(collection) -> str:
    """Generate next available 4-digit user ID."""
    all_data = collection.get(include=["metadatas"])
    ids = []
    if all_data and "metadatas" in all_data:
        for meta in all_data["metadatas"]:
            if isinstance(meta, dict) and "id" in meta:
                try:
                    ids.append(int(meta["id"]))
                except Exception:
                    pass
    next_id = max(ids, default=START_ID - 1) + 1
    if next_id > MAX_ID:
        st.error("User ID limit reached.")
        st.stop()
    return f"{next_id:04d}"


def get_user_by_name(collection, name: str):
    """Look up user metadata by name."""
    results = collection.get(include=["metadatas"])
    if not results or not results.get("metadatas"):
        return None
    for meta in results["metadatas"]:
        if meta.get("name", "").strip().lower() == name.strip().lower():
            return meta
    return None


# ---------------- LOGIN PAGE ----------------
def login_page():
    """Handle login/register and persist login across Streamlit reruns."""
    st.set_page_config(page_title="Login - Autism Support Assistant", page_icon="🔐", layout="centered")

    # If already logged in, skip login UI
    if "user_data" in st.session_state and st.session_state["user_data"]:
        return st.session_state["user_data"]

    st.title("🔐 Welcome to Autism Support Assistant")

    collection = get_user_collection()
    mode = st.radio("Choose an option:", ["Log in (Existing User)", "Register (New User)"], horizontal=True)

    if mode == "Log in (Existing User)":
        name = st.text_input("Enter your name to continue:", key="login_name")
        if name and st.session_state.get("login_trigger", False):
            # triggered by Enter key or button
            user_data = get_user_by_name(collection, name)
            if not user_data:
                st.error("User not found. Please register as a new user.")
                st.session_state["login_trigger"] = False
                st.stop()
            else:
                st.success(f"Welcome back, {user_data['name']} (ID: {user_data['id']})!")
                st.session_state["user_data"] = user_data
                st.session_state["login_trigger"] = False
                return user_data

        # Pressing Enter or clicking button both trigger login
        if st.button("Log In") or (name and st.session_state.get("login_enter_pressed", False)):
            st.session_state["login_trigger"] = True
            st.rerun()

    elif mode == "Register (New User)":
        name = st.text_input("Enter your name:", key="reg_name")
        age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
        has_autistic_child = st.toggle("Have Autistic Child?", value=True)

        if name and st.session_state.get("register_trigger", False):
            existing = get_user_by_name(collection, name)
            if existing:
                st.error("This name already exists. Try logging in instead.")
                st.session_state["register_trigger"] = False
                st.stop()

            user_id = generate_user_id(collection)
            user_metadata = {
                "id": user_id,
                "name": name.strip(),
                "age": int(age),
                "has_autistic_child": 0 if has_autistic_child else 1,  # 0=yes, 1=no
            }

            collection.add(
                ids=[user_id],
                documents=["user_profile"],
                metadatas=[user_metadata],
            )

            st.success(f"Registration successful! Welcome {name}. Your User ID: {user_id}")
            st.session_state["user_data"] = user_metadata
            st.session_state["register_trigger"] = False
            return user_metadata

        if st.button("Register") or (name and st.session_state.get("register_enter_pressed", False)):
            st.session_state["register_trigger"] = True
            st.rerun()

    return None