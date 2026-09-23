"""
MORTGAGE UNDERWRITING APP
Tab 1 — Knowledge Base: Underwriter manages guideline rules (add / update / delete)
Tab 2 — Loan Applications: Upload evidence → gap analysis → Fannie Mae vs FHA recommendation
                           → upload more → re-analyse until gaps are closed

Connects to TWO Qdrant collections:
  • Mortgage             — raw guideline PDFs uploaded by the underwriter
  • CuratedKnowledge_mortgage — pre-curated rules (read-only reference)
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
# DEFAULT API KEYS — edit these so you don't have to type them every time.
# They can still be overridden in the sidebar at runtime.
# ============================================================================
DEFAULT_GROQ_KEY   = os.getenv("GROQ_API_KEY",  "YOUR_GROQ_API_KEY_HERE")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL",    "YOUR_QDRANT_URL_HERE")
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY",    "YOUR_QDRANT_API_KEY_HERE")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Groq model: openai/gpt-oss-120b (replaced llama-3.3-70b-versatile which
# was removed from free/Developer tiers on 2026-08-16).
# Alternative: "openai/gpt-oss-20b" — faster & cheaper, slightly less capable.
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Mortgage Underwriting System")
st.caption("Knowledge Base Management  •  Loan Evidence Gap Analysis  •  Fannie Mae vs FHA Recommendation")

# ============================================================================
# SIDEBAR — API KEY INPUTS (pre-filled with defaults, overrideable)
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

APPLICATIONS_DIR = Path("loan_applications")
APPLICATIONS_DIR.mkdir(exist_ok=True)
KB_REGISTRY = Path("kb_registry.json")

# ============================================================================
# COLLECTION NAMES
# ============================================================================
COLLECTION_RAW     = "Mortgage"
COLLECTION_CURATED = "CuratedKnowledge_mortgage"

# ============================================================================
# GROQ CLIENT (OpenAI-compatible)
# ============================================================================
@st.cache_resource
def get_groq_client(_key):
    return OpenAI(api_key=_key, base_url=GROQ_BASE_URL)

groq_client = get_groq_client(GROQ_KEY)

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    """Unified LLM call via Groq's OpenAI-compatible endpoint."""
    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
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
    """Create payload indexes required for filtered search."""
    for field in ("metadata.guideline", "metadata.source"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

try:
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_KEY)
    vs_raw        = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_RAW)
    ensure_payload_indexes(qdrant_client, COLLECTION_RAW)
    st.sidebar.success(f"✅ Connected to **{COLLECTION_RAW}**")
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
# KB REGISTRY — track which documents are in the knowledge base
# ============================================================================
def load_kb_registry() -> dict:
    if KB_REGISTRY.exists():
        with open(KB_REGISTRY) as f:
            return json.load(f)
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
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, source: str, guideline: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    return [
        Document(page_content=c, metadata={"source": source, "guideline": guideline})
        for c in chunks
    ]

# ============================================================================
# APPLICATION PERSISTENCE
# ============================================================================
def generate_application_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d")
    short = uuid.uuid4().hex[:6].upper()
    return f"APP-{ts}-{short}"

def save_application(app: dict):
    fp = APPLICATIONS_DIR / f"{app['application_id']}.json"
    serializable = {k: v for k, v in app.items() if k != "_file_bytes"}
    with open(fp, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

def load_application(app_id: str) -> dict | None:
    fp = APPLICATIONS_DIR / f"{app_id}.json"
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None

def list_applications() -> list[dict]:
    apps = []
    for fp in sorted(APPLICATIONS_DIR.glob("APP-*.json"), reverse=True):
        with open(fp) as f:
            data = json.load(f)
            apps.append({
                "Application ID": data.get("application_id", fp.stem),
                "Created": data.get("created_at", "—"),
                "Files": len(data.get("evidence_files", [])),
                "Analyses": len(data.get("analysis_history", [])),
                "Status": data.get("status", "Open"),
            })
    return apps

# ============================================================================
# LANGGRAPH — GAP ANALYSIS PIPELINE
# ============================================================================
class AnalysisState(TypedDict):
    evidence_text: str
    fnma_rules: str
    fha_rules: str
    curated_rules: str
    gap_report: str
    recommendation: str

def retrieve_rules(state: AnalysisState) -> dict:
    """Retrieve relevant rules from both Qdrant collections for both guidelines."""
    query = state["evidence_text"][:2000]

    # --- RAW collection: filtered by guideline ---
    fnma_docs, fha_docs = [], []
    for guideline, doc_list in [("Fannie Mae", fnma_docs), ("FHA", fha_docs)]:
        try:
            results = vs_raw.similarity_search(
                query, k=8,
                filter=Filter(must=[
                    FieldCondition(key="metadata.guideline", match=MatchValue(value=guideline))
                ]),
            )
            doc_list.extend(results)
        except Exception:
            pass

    # --- CURATED collection: unfiltered (rules are pre-tagged) ---
    curated_docs = []
    if vs_curated:
        try:
            curated_docs = vs_curated.similarity_search(query, k=10)
        except Exception:
            pass

    return {
        "fnma_rules":    "\n\n".join(d.page_content for d in fnma_docs)    or "No Fannie Mae rules found in knowledge base.",
        "fha_rules":     "\n\n".join(d.page_content for d in fha_docs)     or "No FHA rules found in knowledge base.",
        "curated_rules": "\n\n".join(d.page_content for d in curated_docs) or "No curated rules available.",
    }

def analyse_gaps(state: AnalysisState) -> dict:
    """LLM compares evidence against both guideline sets."""
    system = """You are an expert mortgage underwriter. Compare the borrower's evidence
against BOTH Fannie Mae and FHA guidelines provided.

Produce a structured report with these sections:
## Fannie Mae Assessment
### ✅ Requirements Met
### ⚠️ Partially Met (evidence exists but incomplete)
### ❌ Gaps — Missing Evidence
### 📋 Documents Still Needed

## FHA Assessment
### ✅ Requirements Met
### ⚠️ Partially Met
### ❌ Gaps — Missing Evidence
### 📋 Documents Still Needed

Be specific: name exact documents, amounts, or time periods missing.
Reference the guideline rules that require each item."""

    user = f"""=== BORROWER EVIDENCE ===
{state['evidence_text']}

=== FANNIE MAE RULES ===
{state['fnma_rules']}

=== FHA RULES ===
{state['fha_rules']}

=== CURATED AUTHORITATIVE RULES ===
{state['curated_rules']}"""

    report = call_llm(system, user)
    return {"gap_report": report}

def recommend_guideline(state: AnalysisState) -> dict:
    """LLM recommends Fannie Mae vs FHA based on the gap analysis."""
    system = """You are a senior mortgage advisor. Based on the gap analysis,
recommend whether Fannie Mae or FHA is the BETTER path to approval for this borrower.

Structure your response as:
## 🏆 Recommendation: [Fannie Mae / FHA]

### Comparison Table
| Factor | Fannie Mae | FHA |
|--------|-----------|-----|
| Gaps remaining | ... | ... |
| Flexibility on income | ... | ... |
| Down payment requirement | ... | ... |
| Mortgage insurance | ... | ... |
| Overall fit | ... | ... |

### Why This Path
(2-3 sentences explaining the decisive factors)

### Prioritised Next Steps
1. ...
2. ...
3. ...

### When to Reconsider the Other Option
(Conditions under which the other guideline would become better)

Be decisive — pick ONE. Do not say "it depends"."""

    user = f"""=== GAP ANALYSIS ===
{state['gap_report']}

=== EVIDENCE SUMMARY ===
{state['evidence_text'][:3000]}"""

    rec = call_llm(system, user)
    return {"recommendation": rec}

# Build LangGraph
workflow = StateGraph(AnalysisState)
workflow.add_node("retrieve_rules", retrieve_rules)
workflow.add_node("analyse_gaps", analyse_gaps)
workflow.add_node("recommend_guideline", recommend_guideline)
workflow.set_entry_point("retrieve_rules")
workflow.add_edge("retrieve_rules", "analyse_gaps")
workflow.add_edge("analyse_gaps", "recommend_guideline")
workflow.add_edge("recommend_guideline", END)
analysis_pipeline = workflow.compile()

# ============================================================================
# TAB 1 — KNOWLEDGE BASE
# ============================================================================
tab_kb, tab_app = st.tabs(["📚 Knowledge Base", "📝 Loan Applications"])

with tab_kb:
    st.header("📚 Guideline Knowledge Base")
    st.markdown("Upload, update, or delete mortgage guideline PDFs. These rules are used by **every** loan assessment.")

    # ---- Current documents ----
    registry = load_kb_registry()
    if registry:
        st.subheader("📄 Current Documents")
        for doc_name, info in registry.items():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            col1.write(f"**{doc_name}**")
            col2.write(f"🏷️ {info['guideline']}")
            col3.write(f"🧩 {info['chunk_count']} chunks")
            if col4.button("🗑️ Delete", key=f"del_{doc_name}"):
                point_ids = unregister_kb_document(doc_name)
                if point_ids:
                    try:
                        qdrant_client.delete(
                            collection_name=COLLECTION_RAW,
                            points_selector=PointIdsList(points=point_ids),
                        )
                        st.success(f"Deleted **{doc_name}** ({len(point_ids)} vectors removed)")
                    except Exception as e:
                        st.error(f"Vector deletion failed: {e}")
                st.rerun()
        st.divider()

    # Show curated collection info
    if vs_curated:
        try:
            curated_info = qdrant_client.get_collection(COLLECTION_CURATED)
            st.metric("Curated Rules Collection", f"{curated_info.points_count} vectors", help="Read-only authoritative rules")
        except Exception:
            pass

    # ---- Upload new / update ----
    st.subheader("⬆️ Upload Guidelines")
    guideline_type = st.selectbox("Guideline Type", ["Fannie Mae", "FHA"], key="kb_guideline")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="kb_upload"
    )

    if uploaded_files and st.button("📤 Upload to Knowledge Base", type="primary"):
        for uf in uploaded_files:
            file_bytes = uf.read()
            doc_name = uf.name

            # If document already exists → delete old vectors first (update)
            if doc_name in registry:
                old_ids = unregister_kb_document(doc_name)
                if old_ids:
                    try:
                        qdrant_client.delete(
                            collection_name=COLLECTION_RAW,
                            points_selector=PointIdsList(points=old_ids),
                        )
                        st.info(f"♻️ Replacing existing **{doc_name}** ({len(old_ids)} old vectors removed)")
                    except Exception as e:
                        st.warning(f"Could not remove old vectors for {doc_name}: {e}")

            with st.spinner(f"Processing {doc_name}..."):
                text = extract_text_from_pdf(file_bytes)
                if not text.strip():
                    st.warning(f"⚠️ {doc_name} has no extractable text — skipped.")
                    continue

                docs = chunk_text(text, source=doc_name, guideline=guideline_type)

                # Generate deterministic UUIDs from content hash
                point_ids = []
                for i, doc in enumerate(docs):
                    content_hash = hashlib.md5(f"{doc_name}:{i}:{doc.page_content[:200]}".encode()).hexdigest()
                    point_ids.append(str(uuid.UUID(content_hash)))

                vs_raw.add_documents(docs, ids=point_ids)
                register_kb_document(doc_name, guideline_type, point_ids, len(docs))
                st.success(f"✅ **{doc_name}** → {len(docs)} chunks stored as {guideline_type}")

        st.rerun()

# ============================================================================
# TAB 2 — LOAN APPLICATIONS
# ============================================================================
with tab_app:
    st.header("📝 Loan Applications")

    # ---- Session state init ----
    if "current_app" not in st.session_state:
        st.session_state.current_app = None

    # ---- Open previous / start new ----
    col_new, col_open = st.columns(2)

    with col_new:
        if st.button("🆕 Start New Application", type="primary"):
            new_id = generate_application_id()
            st.session_state.current_app = {
                "application_id": new_id,
                "created_at": datetime.datetime.now().isoformat(),
                "evidence_files": [],
                "evidence_text": "",
                "analysis_history": [],
                "status": "Open",
            }
            save_application(st.session_state.current_app)
            st.success(f"Created **{new_id}**")
            st.rerun()

    with col_open:
        apps = list_applications()
        if apps:
            with st.expander("📂 Open a Previous Application"):
                import pandas as pd
                df = pd.DataFrame(apps)
                st.dataframe(df, use_container_width=True, hide_index=True)
                selected_id = st.selectbox(
                    "Select application",
                    [a["Application ID"] for a in apps],
                    key="open_app_select",
                )
                if st.button("Open"):
                    loaded = load_application(selected_id)
                    if loaded:
                        st.session_state.current_app = loaded
                        st.success(f"Opened **{selected_id}**")
                        st.rerun()

    st.divider()

    # ---- Active application workspace ----
    app = st.session_state.current_app
    if not app:
        st.info("👆 Start a new application or open an existing one to begin.")
        st.stop()

    st.subheader(f"📋 Application: {app['application_id']}")
    st.caption(f"Created: {app['created_at']}  •  Status: {app['status']}  •  Files: {len(app['evidence_files'])}")

    # ---- Upload evidence ----
    st.markdown("### 📎 Upload Evidence Documents")
    st.caption("Upload pay stubs, tax returns, bank statements, gift letters, employment verification, etc.")
    evidence_files = st.file_uploader(
        "Upload borrower evidence PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"evidence_{app['application_id']}",
    )

    if evidence_files and st.button("📥 Add Evidence to Application"):
        for ef in evidence_files:
            file_bytes = ef.read()
            text = extract_text_from_pdf(file_bytes)
            if text.strip():
                app["evidence_files"].append(ef.name)
                app["evidence_text"] += f"\n\n=== {ef.name} ===\n{text}"
                st.success(f"✅ Added **{ef.name}**")
            else:
                st.warning(f"⚠️ {ef.name} has no extractable text — skipped.")
        save_application(app)
        st.rerun()

    # Show current evidence files
    if app["evidence_files"]:
        st.markdown("**Uploaded evidence:**")
        for i, fname in enumerate(app["evidence_files"], 1):
            st.write(f"  {i}. 📄 {fname}")
    else:
        st.info("No evidence files uploaded yet.")

    st.divider()

    # ---- Run gap analysis ----
    if app["evidence_files"]:
        st.markdown("### 🔍 Gap Analysis & Recommendation")

        if st.button("🚀 Run Gap Analysis", type="primary"):
            if not app["evidence_text"].strip():
                st.error("No evidence text to analyse.")
            else:
                with st.spinner("Retrieving rules from knowledge base..."):
                    try:
                        result = analysis_pipeline.invoke({
                            "evidence_text": app["evidence_text"],
                            "fnma_rules": "",
                            "fha_rules": "",
                            "curated_rules": "",
                            "gap_report": "",
                            "recommendation": "",
                        })

                        analysis_entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "files_at_analysis": list(app["evidence_files"]),
                            "gap_report": result["gap_report"],
                            "recommendation": result["recommendation"],
                        }
                        app["analysis_history"].append(analysis_entry)
                        save_application(app)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    # ---- Display latest analysis ----
    if app.get("analysis_history"):
        latest = app["analysis_history"][-1]
        st.markdown(f"**Latest Analysis** — {latest['timestamp']}")
        st.markdown(f"*Based on {len(latest['files_at_analysis'])} files: {', '.join(latest['files_at_analysis'])}*")

        report_tab, rec_tab, hist_tab = st.tabs(["📊 Gap Report", "🏆 Recommendation", "📜 History"])

        with report_tab:
            st.markdown(latest["gap_report"])

        with rec_tab:
            st.markdown(latest["recommendation"])

        with hist_tab:
            for i, entry in enumerate(reversed(app["analysis_history"]), 1):
                with st.expander(f"Analysis #{len(app['analysis_history']) - i + 1} — {entry['timestamp']}"):
                    st.markdown(f"**Files:** {', '.join(entry['files_at_analysis'])}")
                    st.markdown("---")
                    st.markdown(entry["gap_report"])
                    st.markdown("---")
                    st.markdown(entry["recommendation"])

        # ---- Prompt to upload more ----
        st.divider()
        st.markdown("### 🔄 Close the Gaps")
        st.info("Upload additional documents above to address the identified gaps, then re-run the analysis.")
