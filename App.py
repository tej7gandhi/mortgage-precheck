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
from groq import Groq

# ============================================================================
# DEFAULT API KEYS — edit these so you don't have to type them every time.
# They can still be overridden in the sidebar at runtime.
# ============================================================================
DEFAULT_GROQ_KEY   = os.getenv("GROQ_API_KEY",  "YOUR_GROQ_API_KEY_HERE")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL",    "YOUR_QDRANT_URL_HERE")
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY",    "YOUR_QDRANT_API_KEY_HERE")

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
    help="Used for LLM calls (Llama-3.3-70b)"
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
    """Create payload indexes required for filtered search.
    Qdrant needs a 'keyword' index on any field used in a Filter.
    This is idempotent — if the index already exists, Qdrant ignores the call.
    """
    for field in ("metadata.guideline", "metadata.source"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index may already exist, or collection may not exist yet — both are fine
            pass

try:
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_KEY)
    vs_raw        = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_RAW)
    # Ensure required indexes exist so filtered search never 400s
    ensure_payload_indexes(qdrant_client, COLLECTION_RAW)
    st.sidebar.success(f"✅ Connected to **{COLLECTION_RAW}**")
except Exception as e:
    st.sidebar.error(f"❌ {COLLECTION_RAW} connection failed: {e}")
    st.stop()

# Curated collection — optional, don't block if it doesn't exist
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
    for fp in sorted(APPLICATIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(fp) as f:
                d = json.load(f)
            apps.append({
                "Application ID": d.get("application_id", fp.stem),
                "Created": d.get("created_at", "—"),
                "Status": d.get("status", "Draft"),
                "Documents": len(d.get("evidence_files", [])),
                "Open Gaps": d.get("open_gaps", "—"),
                "Recommended": d.get("recommended_guideline", "—"),
            })
        except Exception:
            pass
    return apps

# ============================================================================
# GROQ LLM HELPER
# ============================================================================
def llm_call(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    client = Groq(api_key=GROQ_KEY)
    kwargs = dict(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

# ============================================================================
# LANGGRAPH — GAP ANALYSIS + RECOMMENDATION
# ============================================================================
class AnalysisState(TypedDict):
    evidence_text: str
    fannie_rules: str
    fha_rules: str
    curated_rules: str
    gap_report: str
    recommendation: str

def retrieve_rules(state: AnalysisState) -> dict:
    """Search both collections for Fannie Mae and FHA rules relevant to the evidence."""
    evidence = state["evidence_text"][:2000]  # truncate for query

    # --- Raw collection: filtered by guideline ---
    fannie_docs, fha_docs = [], []
    try:
        fannie_docs = vs_raw.similarity_search(
            evidence, k=8,
            filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="Fannie Mae"))]),
        )
    except Exception:
        pass
    try:
        fha_docs = vs_raw.similarity_search(
            evidence, k=8,
            filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="FHA"))]),
        )
    except Exception:
        pass

    # --- Curated collection: unfiltered (contains both guidelines) ---
    curated_docs = []
    if vs_curated:
        try:
            curated_docs = vs_curated.similarity_search(evidence, k=10)
        except Exception:
            pass

    def fmt(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        ) or "(no rules found)"

    return {
        "fannie_rules": fmt(fannie_docs),
        "fha_rules": fmt(fha_docs),
        "curated_rules": fmt(curated_docs),
    }

def analyse_gaps(state: AnalysisState) -> dict:
    """LLM compares evidence against both guideline sets."""
    system = """You are a senior mortgage underwriter. Analyse the borrower's evidence
against BOTH Fannie Mae and FHA guidelines. Use the curated rules as authoritative
references and the raw rules for additional detail.

Produce a structured markdown report with these sections:
## Fannie Mae Assessment
### ✅ Requirements Met
### ⚠️ Partially Met
### ❌ Gaps — Missing Evidence
## FHA Assessment
### ✅ Requirements Met
### ⚠️ Partially Met
### ❌ Gaps — Missing Evidence

For each gap, specify EXACTLY what document or information is missing.
Be specific and cite which rule/guideline requires it."""

    user = f"""=== BORROWER EVIDENCE ===
{state['evidence_text'][:6000]}

=== FANNIE MAE RULES (Raw) ===
{state['fannie_rules'][:4000]}

=== FHA RULES (Raw) ===
{state['fha_rules'][:4000]}

=== CURATED AUTHORITATIVE RULES ===
{state['curated_rules'][:4000]}"""

    report = llm_call(system, user)
    return {"gap_report": report}

def recommend_guideline(state: AnalysisState) -> dict:
    """LLM recommends Fannie Mae vs FHA based on the gap analysis."""
    system = """Based on the gap analysis, recommend whether the borrower should pursue
Fannie Mae or FHA. Be decisive. Structure your response as:

## 🏆 Recommendation: [Fannie Mae / FHA]
### Why
(2-3 sentence summary)
### Comparison Table
| Criteria | Fannie Mae | FHA |
|----------|-----------|-----|
(fill in key comparison points: gaps remaining, flexibility, mortgage insurance, down payment, etc.)
### Prioritised Next Steps
(numbered list of what to submit next to close gaps under the recommended guideline)
### When to Reconsider
(brief note on when the other guideline might become the better choice)"""

    user = f"""=== GAP ANALYSIS ===
{state['gap_report']}"""

    rec = llm_call(system, user)

    # Extract the recommended guideline name for the application record
    recommended = "Undetermined"
    first_line = rec.split("\n")[0].lower()
    if "fannie" in first_line:
        recommended = "Fannie Mae"
    elif "fha" in first_line:
        recommended = "FHA"

    return {"recommendation": rec, "_recommended_name": recommended}

# Build the LangGraph
workflow = StateGraph(AnalysisState)
workflow.add_node("retrieve_rules", retrieve_rules)
workflow.add_node("analyse_gaps", analyse_gaps)
workflow.add_node("recommend_guideline", recommend_guideline)
workflow.set_entry_point("retrieve_rules")
workflow.add_edge("retrieve_rules", "analyse_gaps")
workflow.add_edge("analyse_gaps", "recommend_guideline")
workflow.add_edge("recommend_guideline", END)
analysis_chain = workflow.compile()

# ============================================================================
# TAB LAYOUT
# ============================================================================
tab_kb, tab_app = st.tabs(["📚 Knowledge Base", "📝 Loan Applications"])

# ============================  TAB 1: KB  ===================================
with tab_kb:
    st.header("📚 Knowledge Base Management")
    st.markdown("Upload, update, or delete mortgage guideline documents. These rules are used to assess **every** loan application.")

    # --- Current KB contents ---
    registry = load_kb_registry()
    if registry:
        st.subheader("📄 Current Documents")
        for doc_name, info in registry.items():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            col1.write(f"**{doc_name}**")
            col2.write(f"🏷️ {info['guideline']}")
            col3.write(f"📦 {info['chunk_count']} chunks")
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
                        st.error(f"Failed to delete vectors: {e}")
                st.rerun()
    else:
        st.info("Knowledge base is empty. Upload guideline documents below.")

    # Show curated collection stats if available
    if vs_curated:
        try:
            curated_info = qdrant_client.get_collection(COLLECTION_CURATED)
            st.metric("Curated Rules Collection", f"{curated_info.points_count} vectors", help="Read-only authoritative rules")
        except Exception:
            pass

    # --- Upload new / update existing ---
    st.subheader("⬆️ Upload Guideline Document")
    st.caption("If you upload a file with the same name as an existing document, it will be **replaced** automatically.")

    guideline_type = st.selectbox("Guideline Type", ["Fannie Mae", "FHA"], key="kb_guideline")
    uploaded_kb = st.file_uploader("Upload PDF", type=["pdf"], key="kb_upload", accept_multiple_files=True)

    if uploaded_kb and st.button("📥 Add to Knowledge Base", type="primary"):
        for uf in uploaded_kb:
            with st.spinner(f"Processing {uf.name}..."):
                # If doc already exists, delete old vectors first
                if uf.name in registry:
                    old_ids = unregister_kb_document(uf.name)
                    if old_ids:
                        try:
                            qdrant_client.delete(
                                collection_name=COLLECTION_RAW,
                                points_selector=PointIdsList(points=old_ids),
                            )
                        except Exception:
                            pass
                    st.info(f"♻️ Replacing existing **{uf.name}**")

                text = extract_text_from_pdf(uf.read())
                docs = chunk_text(text, source=uf.name, guideline=guideline_type)

                # Generate deterministic UUIDs for tracking
                point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{uf.name}-{i}")) for i in range(len(docs))]

                vs_raw.add_documents(docs, ids=point_ids)
                register_kb_document(uf.name, guideline_type, point_ids, len(docs))
                st.success(f"✅ **{uf.name}** — {len(docs)} chunks indexed under {guideline_type}")
        st.rerun()

# ==========================  TAB 2: APPLICATIONS  ===========================
with tab_app:
    st.header("📝 Loan Applications")

    # --- Session state init ---
    if "current_app" not in st.session_state:
        st.session_state.current_app = None

    # --- Open previous or start new ---
    col_new, col_open = st.columns(2)
    with col_new:
        if st.button("🆕 Start New Application", type="primary"):
            app_id = generate_application_id()
            st.session_state.current_app = {
                "application_id": app_id,
                "created_at": datetime.datetime.now().isoformat(),
                "status": "Draft",
                "evidence_files": [],
                "evidence_text": "",
                "analysis_history": [],
                "open_gaps": "—",
                "recommended_guideline": "—",
            }
            save_application(st.session_state.current_app)
            st.success(f"Created **{app_id}**")

    with col_open:
        with st.expander("📂 Open a Previous Application"):
            apps = list_applications()
            if apps:
                import pandas as pd
                st.dataframe(pd.DataFrame(apps), use_container_width=True, hide_index=True)
                sel_id = st.selectbox("Select Application ID", [a["Application ID"] for a in apps])
                if st.button("Open"):
                    loaded = load_application(sel_id)
                    if loaded:
                        st.session_state.current_app = loaded
                        st.rerun()
            else:
                st.info("No applications yet.")

    st.divider()

    # --- Active application workspace ---
    app = st.session_state.current_app
    if not app:
        st.info("👆 Start a new application or open an existing one.")
        st.stop()

    # Header
    c1, c2, c3 = st.columns(3)
    c1.metric("Application ID", app["application_id"])
    c2.metric("Status", app["status"])
    c3.metric("Evidence Files", len(app["evidence_files"]))

    # --- Upload evidence ---
    st.subheader("📎 Upload Evidence Documents")
    st.caption("Upload pay stubs, tax returns, bank statements, gift letters, etc.")
    evidence_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key=f"ev_{app['application_id']}"
    )

    if evidence_files and st.button("📥 Add Evidence to Application"):
        new_text_parts = []
        for ef in evidence_files:
            with st.spinner(f"Reading {ef.name}..."):
                text = extract_text_from_pdf(ef.read())
                new_text_parts.append(f"=== {ef.name} ===\n{text}")
                if ef.name not in app["evidence_files"]:
                    app["evidence_files"].append(ef.name)
        app["evidence_text"] = app.get("evidence_text", "") + "\n\n".join(new_text_parts)
        app["status"] = "Evidence Uploaded"
        save_application(app)
        st.success(f"Added {len(evidence_files)} file(s). Total evidence: {len(app['evidence_files'])} documents.")
        st.rerun()

    # Show current evidence list
    if app["evidence_files"]:
        st.write("**📄 Evidence on file:**", ", ".join(app["evidence_files"]))

    st.divider()

    # --- Run analysis ---
    st.subheader("🔍 Gap Analysis & Recommendation")
    if not app.get("evidence_text"):
        st.info("Upload evidence documents above, then run the analysis.")
    else:
        if st.button("🚀 Run Gap Analysis", type="primary"):
            with st.spinner("Retrieving rules from knowledge base..."):
                try:
                    result = analysis_chain.invoke({
                        "evidence_text": app["evidence_text"],
                        "fannie_rules": "",
                        "fha_rules": "",
                        "curated_rules": "",
                        "gap_report": "",
                        "recommendation": "",
                    })

                    # Count open gaps
                    gap_count = result["gap_report"].lower().count("❌")
                    app["open_gaps"] = f"{gap_count} gap(s)" if gap_count else "✅ None"
                    app["status"] = "Complete — No Gaps" if gap_count == 0 else "Gaps Identified"

                    # Extract recommendation name
                    rec_name = "Undetermined"
                    first_line = result["recommendation"].split("\n")[0].lower()
                    if "fannie" in first_line:
                        rec_name = "Fannie Mae"
                    elif "fha" in first_line:
                        rec_name = "FHA"
                    app["recommended_guideline"] = rec_name

                    # Save analysis run
                    run = {
                        "run_at": datetime.datetime.now().isoformat(),
                        "evidence_count": len(app["evidence_files"]),
                        "gap_report": result["gap_report"],
                        "recommendation": result["recommendation"],
                        "open_gaps": app["open_gaps"],
                    }
                    app.setdefault("analysis_history", []).append(run)
                    save_application(app)

                    st.success("Analysis complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # --- Display latest results ---
    history = app.get("analysis_history", [])
    if history:
        latest = history[-1]
        st.markdown("---")
        st.markdown("### 📋 Gap Analysis Report")
        st.markdown(latest["gap_report"])
        st.markdown("---")
        st.markdown("### 🏆 Guideline Recommendation")
        st.markdown(latest["recommendation"])

        # Show history of runs
        if len(history) > 1:
            with st.expander(f"📜 Analysis History ({len(history)} runs)"):
                for i, run in enumerate(reversed(history), 1):
                    st.markdown(f"**Run {len(history) - i + 1}** — {run['run_at']}  |  Evidence: {run['evidence_count']} files  |  Gaps: {run['open_gaps']}")
                    if st.checkbox(f"Show details", key=f"hist_{i}"):
                        st.markdown(run["gap_report"])
                        st.markdown(run["recommendation"])
                    st.divider()

    # --- Upload more to close gaps ---
    if history and app["status"] != "Complete — No Gaps":
        st.markdown("---")
        st.subheader("📎 Upload Additional Documents to Close Gaps")
        st.caption("Upload the missing documents identified above, then re-run the analysis.")
        more_files = st.file_uploader(
            "Upload more PDFs", type=["pdf"], accept_multiple_files=True, key=f"more_{app['application_id']}_{len(history)}"
        )
        if more_files and st.button("📥 Add & Re-analyse"):
            new_parts = []
            for ef in more_files:
                with st.spinner(f"Reading {ef.name}..."):
                    text = extract_text_from_pdf(ef.read())
                    new_parts.append(f"=== {ef.name} ===\n{text}")
                    if ef.name not in app["evidence_files"]:
                        app["evidence_files"].append(ef.name)
            app["evidence_text"] += "\n\n" + "\n\n".join(new_parts)
            save_application(app)
            st.rerun()
