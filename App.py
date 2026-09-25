"""
MORTGAGE UNDERWRITING APP - FIXED VERSION
Fixes:
 1. NameError: load_kb_registry not defined (functions now defined BEFORE usage)
 2. Qdrant client dependency in registry functions (now param-based)
"""

import streamlit as st
import os
import io
import json
import uuid
import hashlib
import datetime
from pathlib import Path
from typing import TypedDict, List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PointIdsList,
    PayloadSchemaType,
)
from openai import OpenAI

# ============================================================================
# DEFAULT API KEYS
# ============================================================================
DEFAULT_GROQ_KEY   = os.getenv("GROQ_API_KEY",  "YOUR_GROQ_API_KEY_HERE")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL",    "YOUR_QDRANT_URL_HERE")
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY",    "YOUR_QDRANT_API_KEY_HERE")

# ============================================================================
# MODEL CONFIG
# ============================================================================
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Mortgage Underwriting System")
st.caption("Knowledge Base Management  •  Loan Evidence Gap Analysis  •  Fannie Mae vs FHA Recommendation")

# ============================================================================
# CONSTANTS - Defined early
# ============================================================================
APPLICATIONS_DIR = Path("loan_applications")
APPLICATIONS_DIR.mkdir(exist_ok=True)
KB_REGISTRY = Path("kb_registry.json")

COLLECTION_RAW     = "Mortgage"
COLLECTION_CURATED = "CuratedKnowledge_mortgage"

# ============================================================================
# SIDEBAR — API KEY INPUTS
# ============================================================================
st.sidebar.header("🔑 API Configuration")
st.sidebar.caption("Pre-filled from defaults. Override here if needed.")

GROQ_KEY = st.sidebar.text_input(
    "Groq API Key", value=DEFAULT_GROQ_KEY, type="password",
    help="Used for LLM calls (GPT-OSS-120B via Groq)"
)
QDRANT_URL = st.sidebar.text_input(
    "Qdrant URL", value=DEFAULT_QDRANT_URL,
    help="Your Qdrant Cloud cluster URL"
)
QDRANT_KEY = st.sidebar.text_input(
    "Qdrant API Key", value=DEFAULT_QDRANT_KEY, type="password",
    help="Qdrant Cloud API key"
)

placeholder_vals = {"YOUR_GROQ_API_KEY_HERE", "YOUR_QDRANT_URL_HERE", "YOUR_QDRANT_API_KEY_HERE", ""}
if GROQ_KEY in placeholder_vals or QDRANT_URL in placeholder_vals or QDRANT_KEY in placeholder_vals:
    st.sidebar.warning("⚠️ One or more API keys are still placeholders. Please set them above or edit the defaults in the source code.")
    st.stop()

# ============================================================================
# KB REGISTRY — DEFINED BEFORE USAGE (THIS FIXES YOUR ERROR)
# ============================================================================
def rebuild_kb_registry_from_qdrant(client: QdrantClient, collection_name: str) -> dict:
    """Rebuild registry by scrolling Qdrant collection - Qdrant is source of truth."""
    try:
        registry = {}
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                meta = payload.get("metadata", {}) or payload
                source = meta.get("source") or payload.get("source") or "unknown.pdf"
                guideline = meta.get("guideline") or payload.get("guideline") or "Unknown"
                source = str(source)

                if source not in registry:
                    registry[source] = {
                        "guideline": guideline,
                        "point_ids": [],
                        "chunk_count": 0,
                        "uploaded_at": datetime.datetime.now().isoformat(),
                        "rebuilt": True,
                    }
                registry[source]["point_ids"].append(str(p.id))
                registry[source]["chunk_count"] += 1
                if guideline != "Unknown":
                    registry[source]["guideline"] = guideline

            if next_offset is None:
                break
            offset = next_offset

        if registry:
            with open(KB_REGISTRY, "w") as f:
                json.dump(registry, f, indent=2, default=str)
        return registry
    except Exception as e:
        print(f"rebuild failed: {e}")
        return {}

def load_kb_registry(client: QdrantClient = None, collection_name: str = COLLECTION_RAW) -> dict:
    """Load registry - if local file missing/empty, rebuild from Qdrant."""
    if KB_REGISTRY.exists():
        try:
            with open(KB_REGISTRY) as f:
                data = json.load(f)
                if data:
                    return data
        except Exception:
            pass
    # fallback rebuild if client provided
    if client is not None:
        return rebuild_kb_registry_from_qdrant(client, collection_name)
    return {}

def save_kb_registry(registry: dict):
    with open(KB_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2, default=str)

def register_kb_document(doc_name: str, guideline: str, point_ids: list[str], chunk_count: int):
    registry = load_kb_registry()
    registry[doc_name] = {
        "guideline": guideline,
        "point_ids": point_ids,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.datetime.now().isoformat(),
    }
    save_kb_registry(registry)

def unregister_kb_document(doc_name: str) -> list[str]:
    registry = load_kb_registry()
    entry = registry.pop(doc_name, None)
    save_kb_registry(registry)
    return entry["point_ids"] if entry else []

# ============================================================================
# PDF HELPERS
# ============================================================================
def extract_pdf_text(file_bytes: bytes, source_name: str = "upload") -> list[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": source_name, "page": i + 1}))
    return docs

def chunk_documents(docs: list[Document], chunk_size: int = 800, overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# ============================================================================
# APPLICATION PERSISTENCE
# ============================================================================
def generate_app_id() -> str:
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"APP-{date_str}-{short_uuid}"

def save_application(app_data: dict):
    app_id = app_data["app_id"]
    path = APPLICATIONS_DIR / f"{app_id}.json"
    with open(path, "w") as f:
        json.dump(app_data, f, indent=2, default=str)

def load_application(app_id: str) -> dict | None:
    path = APPLICATIONS_DIR / f"{app_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def list_applications() -> list[dict]:
    apps = []
    for f in sorted(APPLICATIONS_DIR.glob("APP-*.json"), reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
                apps.append({
                    "app_id": data.get("app_id", f.stem),
                    "created": data.get("created_at", "Unknown"),
                    "documents": len(data.get("evidence_files", [])),
                    "analyses": len(data.get("analysis_history", [])),
                    "status": data.get("status", "Open"),
                })
        except:
            continue
    return apps

# ============================================================================
# GROQ CLIENT (OpenAI-compatible)
# ============================================================================
@st.cache_resource
def get_groq_client(_key):
    return OpenAI(api_key=_key, base_url=GROQ_BASE_URL)

groq_client = get_groq_client(GROQ_KEY)

def truncate_text(text: str, max_chars: int) -> str:
    """Truncate to max_chars to stay under Groq TPM limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for token limit]"

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1500,  # reduced to stay under TPM
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = groq_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

# ============================================================================
# QDRANT & EMBEDDINGS
# ============================================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client(_url, _key):
    return QdrantClient(url=_url, api_key=_key)

@st.cache_resource
def get_vector_store(_url, _key, _collection):
    emb = get_embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=emb,
        collection_name=_collection,
        url=_url,
        api_key=_key,
    )

def ensure_payload_indexes(client: QdrantClient, collection_name: str):
    for field in ("metadata.guideline", "metadata.source"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

# ============================================================================
# NOW CONNECT TO QDRANT (functions already defined, so no NameError)
# ============================================================================
try:
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_KEY)
    vs_raw        = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_RAW)
    ensure_payload_indexes(qdrant_client, COLLECTION_RAW)
    # Auto-rebuild registry if needed on startup
    _startup_registry = load_kb_registry(client=qdrant_client, collection_name=COLLECTION_RAW)
    st.sidebar.success(f"✅ Connected to **{COLLECTION_RAW}** ({len(_startup_registry)} docs)")
except Exception as e:
    st.sidebar.error(f"❌ {COLLECTION_RAW} connection failed: {e}")
    st.stop()

vs_curated = None
try:
    vs_curated = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_CURATED)
    ensure_payload_indexes(qdrant_client, COLLECTION_CURATED)
    st.sidebar.success(f"✅ Connected to **{COLLECTION_CURATED}**")
except Exception:
    st.sidebar.info(f"ℹ️ {COLLECTION_CURATED} not found — using raw collection only")

# ============================================================================
# LANGGRAPH STATE & PIPELINE
# ============================================================================
class UnderwritingState(TypedDict):
    evidence_text: str
    evidence_files: List[str]
    fnma_rules: str
    fha_rules: str
    curated_rules: str
    gap_analysis: str
    recommendation: str

def retrieve_rules(state: UnderwritingState) -> dict:
    query = truncate_text(state["evidence_text"], 500)  # reduced from 1500 to 500
    fnma_docs = []
    fha_docs = []
    try:
        fnma_docs = vs_raw.similarity_search(
            query, k=3,  # reduced from 8 to 3
            filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="Fannie Mae"))]),
        )
    except Exception:
        pass
    try:
        fha_docs = vs_raw.similarity_search(
            query, k=3,  # reduced from 8 to 3
            filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="FHA"))]),
        )
    except Exception:
        pass

    fnma_text = "\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{truncate_text(d.page_content, 600)}"
        for d in fnma_docs
    )
    fha_text = "\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{truncate_text(d.page_content, 600)}"
        for d in fha_docs
    )

    curated_text = ""
    if vs_curated:
        try:
            curated_docs = vs_curated.similarity_search(query, k=2)  # reduced from 6 to 2
            curated_text = "\n\n".join(truncate_text(d.page_content, 600) for d in curated_docs)
        except:
            pass

    return {"fnma_rules": truncate_text(fnma_text, 2000), "fha_rules": truncate_text(fha_text, 2000), "curated_rules": truncate_text(curated_text, 1000)}

def analyse_gaps(state: UnderwritingState) -> dict:
    system = """You are a senior mortgage underwriter. Compare borrower evidence against Fannie Mae and FHA guidelines.
List what is present, what is missing, and specific gaps. Be concise due to token limits."""

    user = f"""
FILES: {', '.join(state['evidence_files'])}
EVIDENCE (first 3000 chars):
{truncate_text(state['evidence_text'], 3000)}

FANNIE MAE (first 1500 chars):
{truncate_text(state['fnma_rules'], 1500)}

FHA (first 1500 chars):
{truncate_text(state['fha_rules'], 1500)}

CURATED (first 800 chars):
{truncate_text(state['curated_rules'], 800)}

Provide structured gap analysis:
1. Evidence Summary
2. Fannie Mae Gaps
3. FHA Gaps
4. Critical Missing Docs
"""

    gap_analysis = call_llm(system, user, temperature=0.2)
    return {"gap_analysis": gap_analysis}

def recommend_guideline(state: UnderwritingState) -> dict:
    system = """You are a senior mortgage underwriter. Recommend Fannie Mae vs FHA. Be concise."""

    user = f"""
FILES: {state['evidence_files']}
GAPS: {truncate_text(state['gap_analysis'], 2000)}

FNMA: {truncate_text(state['fnma_rules'], 1000)}
FHA: {truncate_text(state['fha_rules'], 1000)}

Return markdown:
### Recommendation: [Fannie Mae / FHA / Neither]
**Confidence:** High/Med/Low
**Reasoning:** brief
**Next Steps:**
**Risk:**
"""

    recommendation = call_llm(system, user, temperature=0.1)
    return {"recommendation": recommendation}

# Build graph
workflow = StateGraph(UnderwritingState)
workflow.add_node("retrieve_rules", retrieve_rules)
workflow.add_node("analyse_gaps", analyse_gaps)
workflow.add_node("recommend_guideline", recommend_guideline)

workflow.set_entry_point("retrieve_rules")
workflow.add_edge("retrieve_rules", "analyse_gaps")
workflow.add_edge("analyse_gaps", "recommend_guideline")
workflow.add_edge("recommend_guideline", END)

pipeline = workflow.compile()

# ============================================================================
# STREAMLIT UI - TABS
# ============================================================================
tab1, tab2 = st.tabs(["📚 Knowledge Base", "📋 Loan Applications"])

with tab1:
    st.subheader("📚 Guideline Knowledge Base")
    kb_registry = load_kb_registry()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Docs", len(kb_registry))
    col2.metric("Total Chunks", sum(v['chunk_count'] for v in kb_registry.values()) if kb_registry else 0)
    col3.metric("Guidelines", f"{len(set(v['guideline'] for v in kb_registry.values()))} types" if kb_registry else "0")

    st.divider()
    st.subheader("📤 Upload New Guideline")
    guideline_type = st.selectbox("Guideline Type", ["Fannie Mae", "FHA", "Other"])
    uploaded_kb = st.file_uploader("Upload guideline PDFs", type=["pdf"], accept_multiple_files=True, key="kb_uploader")

    if uploaded_kb and st.button("Add to Knowledge Base"):
        for pdf_file in uploaded_kb:
            with st.spinner(f"Processing {pdf_file.name}..."):
                try:
                    docs = extract_pdf_text(pdf_file.read(), source_name=pdf_file.name)
                    chunks = chunk_documents(docs)
                    for c in chunks:
                        c.metadata["guideline"] = guideline_type
                        c.metadata["source"] = pdf_file.name

                    # Add to vector store
                    point_ids = vs_raw.add_documents(chunks)
                    # Handle different return types
                    if isinstance(point_ids, list):
                        str_ids = [str(x) for x in point_ids]
                    else:
                        str_ids = [str(uuid.uuid4()) for _ in chunks]

                    register_kb_document(pdf_file.name, guideline_type, str_ids, len(chunks))
                    st.success(f"✅ {pdf_file.name} added ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Failed {pdf_file.name}: {e}")

    st.divider()
    st.subheader("📄 Existing Documents")
    if kb_registry:
        for doc_name, info in kb_registry.items():
            c1, c2, c3, c4 = st.columns([3,1,1,1])
            c1.write(f"📄 {doc_name}")
            c2.write(f"🏷️ {info['guideline']}")
            c3.write(f"🧩 {info['chunk_count']}")
            if c4.button("Delete", key=f"del_{doc_name}"):
                ids = unregister_kb_document(doc_name)
                if ids:
                    try:
                        qdrant_client.delete(collection_name=COLLECTION_RAW, points_selector=PointIdsList(points=ids))
                    except Exception as e:
                        st.error(f"Qdrant delete failed: {e}")
                st.rerun()
    else:
        st.info("No documents in knowledge base. Upload some above.")

with tab2:
    st.subheader("📋 Loan Applications")

    if "current_app" not in st.session_state:
        st.session_state.current_app = None

    col_new, col_reload = st.columns(2)
    if col_new.button("➕ New Application", use_container_width=True):
        app_id = generate_app_id()
        app_data = {
            "app_id": app_id,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "Open",
            "evidence_files": [],
            "evidence_text": "",
            "analysis_history": []
        }
        save_application(app_data)
        st.session_state.current_app = app_data
        st.rerun()

    if col_reload.button("🔄 Refresh List", use_container_width=True):
        st.rerun()

    apps = list_applications()
    if apps:
        st.write("**Recent Applications:**")
        for app in apps[:10]:
            c1, c2, c3, c4, c5 = st.columns([2,1,1,1,1])
            c1.write(f"**{app['app_id']}**")
            c2.write(f"📅 {app['created'][:10]}")
            c3.write(f"📄 {app['documents']} docs")
            c4.write(f"🔍 {app['analyses']} runs")
            if c5.button("Open", key=f"open_{app['app_id']}"):
                st.session_state.current_app = load_application(app["app_id"])
                st.rerun()

    st.divider()

    app = st.session_state.current_app

    if app is None:
        st.info("👆 Start a new application or open an existing one to begin.", icon="📋")
        st.stop()

    st.subheader(f"📋 Application: {app['app_id']}")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Status", app["status"])
    col_s2.metric("Evidence Files", len(app["evidence_files"]))
    col_s3.metric("Analyses Run", len(app["analysis_history"]))
    col_s4.metric("Created", app["created_at"][:10])

    st.divider()

    st.subheader("📤 Upload Evidence Documents")
    evidence_files = st.file_uploader(
        "Upload evidence PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"evidence_{app['app_id']}",
    )

    if evidence_files:
        new_files_added = False
        for ef in evidence_files:
            if ef.name not in app["evidence_files"]:
                with st.spinner(f"Processing {ef.name}..."):
                    docs = extract_pdf_text(ef.read(), source_name=ef.name)
                    new_text = "\n\n".join(
                        f"--- {ef.name} (Page {d.metadata['page']}) ---\n{d.page_content}"
                        for d in docs
                    )
                    app["evidence_text"] += "\n\n" + new_text
                    app["evidence_files"].append(ef.name)
                    new_files_added = True

        if new_files_added:
            save_application(app)
            st.success(f"✅ {len(evidence_files)} file(s) processed.")

    if app["evidence_files"]:
        with st.expander(f"📂 Evidence Files ({len(app['evidence_files'])})"):
            for i, fname in enumerate(app["evidence_files"], 1):
                st.write(f"{i}. 📄 {fname}")
    else:
        st.warning("No evidence files uploaded yet.")

    st.divider()

    st.subheader("🔍 Run Gap Analysis")
    kb_registry = load_kb_registry()
    if not kb_registry:
        st.error("⚠️ Knowledge Base is empty. Please upload guidelines in the Knowledge Base tab first.")
    elif not app["evidence_files"]:
        st.warning("⚠️ Upload at least one evidence document before running the analysis.")
    else:
        fnma_in_kb = any(v["guideline"] == "Fannie Mae" for v in kb_registry.values())
        fha_in_kb = any(v["guideline"] == "FHA" for v in kb_registry.values())
        st.caption(
            f"Knowledge Base: {'🏛️ Fannie Mae ✅' if fnma_in_kb else '🏛️ Fannie Mae ❌'}"
            f"  |  {'🏠 FHA ✅' if fha_in_kb else '🏠 FHA ❌'}"
            f"  |  📄 {len(kb_registry)} docs  |  🧩 {sum(v['chunk_count'] for v in kb_registry.values())} chunks"
        )

        if st.button("🚀 Run Gap Analysis & Recommendation", use_container_width=True, type="primary"):
            with st.spinner("Analysing evidence against Fannie Mae & FHA guidelines..."):
                try:
                    result = pipeline.invoke({
                        "evidence_text": app["evidence_text"],
                        "evidence_files": app["evidence_files"],
                        "fnma_rules": "",
                        "fha_rules": "",
                        "curated_rules": "",
                        "gap_analysis": "",
                        "recommendation": "",
                    })

                    analysis_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "evidence_count": len(app["evidence_files"]),
                        "evidence_files": list(app["evidence_files"]),
                        "gap_analysis": result["gap_analysis"],
                        "recommendation": result["recommendation"],
                    }
                    app["analysis_history"].append(analysis_entry)
                    save_application(app)
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if app["analysis_history"]:
        latest = app["analysis_history"][-1]

        st.divider()
        st.subheader("📊 Latest Analysis Results")
        st.caption(f"Run #{len(app['analysis_history'])}  •  {latest['timestamp'][:19].replace('T', ' at ')}  •  Based on {latest['evidence_count']} doc(s)")

        with st.container(border=True):
            st.markdown(latest["recommendation"])

        st.divider()

        with st.container(border=True):
            st.markdown("## 📋 Detailed Gap Analysis")
            st.markdown(latest["gap_analysis"])

        st.divider()
        st.info(
            "**📤 Upload additional documents above** to cover the identified gaps, "
            "then click **Run Gap Analysis** again.",
            icon="💡"
        )

        if len(app["analysis_history"]) > 1:
            with st.expander(f"📜 Analysis History ({len(app['analysis_history'])} runs)"):
                for i, entry in enumerate(reversed(app["analysis_history"]), 1):
                    run_num = len(app["analysis_history"]) - i + 1
                    st.markdown(f"### Run #{run_num} — {entry['timestamp'][:19].replace('T', ' at ')}")
                    st.caption(f"Documents: {', '.join(entry['evidence_files'])}")
                    with st.container(border=True):
                        st.markdown(entry["recommendation"])
                    with st.container(border=True):
                        st.markdown(entry["gap_analysis"])
                    st.divider()
