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
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList
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

try:
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_KEY)
    vs_raw        = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_RAW)
    st.sidebar.success(f"✅ Connected to **{COLLECTION_RAW}**")
except Exception as e:
    st.sidebar.error(f"❌ {COLLECTION_RAW} connection failed: {e}")
    st.stop()

# Curated collection — optional, don't block if it doesn't exist
vs_curated = None
try:
    vs_curated = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_CURATED)
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
def call_groq(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
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
    return client.chat.completions.create(**kwargs).choices[0].message.content

# ============================================================================
# GAP ANALYSIS ENGINE (LangGraph) — queries BOTH collections
# ============================================================================
class AnalysisState(TypedDict):
    evidence_text: str
    evidence_files: List[str]
    fnma_raw_rules: str
    fnma_curated_rules: str
    fha_raw_rules: str
    fha_curated_rules: str
    gap_report: str
    recommendation: str
    open_gaps: int
    status: str

def retrieve_rules(state: AnalysisState) -> dict:
    """Search both Qdrant collections for Fannie Mae and FHA rules relevant to this evidence."""
    evidence = state["evidence_text"][:3000]
    query = f"Mortgage underwriting rules and requirements for: {evidence[:500]}"

    # --- RAW collection (filtered by guideline tag) ---
    fnma_filter = Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="Fannie Mae"))])
    fha_filter  = Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="FHA"))])

    fnma_raw_docs = vs_raw.similarity_search(query, k=8, filter=fnma_filter)
    fha_raw_docs  = vs_raw.similarity_search(query, k=8, filter=fha_filter)

    fnma_raw = "\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in fnma_raw_docs
    ) or "No Fannie Mae rules found in raw KB."

    fha_raw = "\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in fha_raw_docs
    ) or "No FHA rules found in raw KB."

    # --- CURATED collection (unfiltered — it's all curated rules) ---
    fnma_curated = ""
    fha_curated  = ""
    if vs_curated is not None:
        try:
            curated_docs = vs_curated.similarity_search(query, k=10)
            for d in curated_docs:
                block = f"[Curated: {d.metadata.get('source', '?')}]\n{d.page_content}\n\n"
                content_lower = d.page_content.lower()
                if "fannie" in content_lower or "fnma" in content_lower or "conventional" in content_lower:
                    fnma_curated += block
                elif "fha" in content_lower:
                    fha_curated += block
                else:
                    # General rule — include in both
                    fnma_curated += block
                    fha_curated += block
        except Exception:
            pass

    return {
        "fnma_raw_rules":     fnma_raw,
        "fnma_curated_rules": fnma_curated or "No curated Fannie Mae rules available.",
        "fha_raw_rules":      fha_raw,
        "fha_curated_rules":  fha_curated or "No curated FHA rules available.",
    }

def analyse_gaps(state: AnalysisState) -> dict:
    """LLM compares evidence against both guideline sets and produces structured gap report."""
    system = """You are an expert mortgage underwriter. You have TWO sources of rules for each guideline:
1. Raw guideline documents uploaded by the underwriter
2. Curated knowledge rules (pre-processed, high-signal)

Treat curated rules as authoritative; use raw rules for additional context and detail.

Analyse the borrower's evidence documents against BOTH Fannie Mae and FHA guidelines.
Produce a structured markdown report with these exact sections:

## 📊 Evidence Summary
Brief summary of what the borrower provided.

## ✅ Fannie Mae — Requirements Met
List items fully satisfied under Fannie Mae rules, with the rule citation.

## ⚠️ Fannie Mae — Gaps / Missing Evidence
List what is MISSING or INSUFFICIENT. Be specific about what document is needed and why.

## ✅ FHA — Requirements Met
List items fully satisfied under FHA rules, with the rule citation.

## ⚠️ FHA — Gaps / Missing Evidence
List what is MISSING or INSUFFICIENT under FHA. Be specific.

## 📋 Consolidated Gap List
A numbered list of ALL unique gaps across both guidelines. Each gap should state:
- What is missing
- Which guideline(s) require it
- What document would satisfy it

Be thorough — check income verification, asset documentation, employment history,
credit requirements, property requirements, gift funds documentation, large deposit
sourcing, and any other applicable underwriting requirements."""

    user = f"""=== BORROWER EVIDENCE DOCUMENTS ===
{state['evidence_text']}

=== FANNIE MAE RULES (Raw KB) ===
{state['fnma_raw_rules']}

=== FANNIE MAE RULES (Curated KB) ===
{state['fnma_curated_rules']}

=== FHA RULES (Raw KB) ===
{state['fha_raw_rules']}

=== FHA RULES (Curated KB) ===
{state['fha_curated_rules']}"""

    report = call_groq(system, user)
    return {"gap_report": report}

def recommend_guideline(state: AnalysisState) -> dict:
    """LLM picks the better guideline path and explains why."""
    system = """You are a senior mortgage underwriter advisor. Based on the gap analysis,
recommend whether the borrower should pursue Fannie Mae (conventional) or FHA.

Structure your response as:

## 🏆 Recommendation: [Fannie Mae / FHA]

### Why
2-3 sentences explaining the decisive reason.

### Comparison Table
| Factor | Fannie Mae | FHA |
|--------|-----------|-----|
| Open Gaps | N | N |
| Easier Path to Close | ... | ... |
| Mortgage Insurance | ... | ... |
| Down Payment Flexibility | ... | ... |
| Overall Fit | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### Next Steps
Numbered list of prioritised actions to close the remaining gaps under the RECOMMENDED guideline.

### When to Reconsider
One sentence on when the other guideline would become the better option.

Also output a single JSON line at the very end in this exact format:
RECOMMENDATION_JSON: {"guideline": "Fannie Mae" or "FHA", "open_gaps": <number>}"""

    user = f"""=== GAP ANALYSIS REPORT ===
{state['gap_report']}"""

    result = call_groq(system, user)

    # Extract structured data
    rec_guideline = "—"
    open_gaps = 0
    for line in result.split("\n"):
        if line.strip().startswith("RECOMMENDATION_JSON:"):
            try:
                jdata = json.loads(line.split("RECOMMENDATION_JSON:")[1].strip())
                rec_guideline = jdata.get("guideline", "—")
                open_gaps = jdata.get("open_gaps", 0)
            except Exception:
                pass
            break

    # Remove the JSON line from display
    clean = "\n".join(
        l for l in result.split("\n") if not l.strip().startswith("RECOMMENDATION_JSON:")
    )

    status = "✅ Complete" if open_gaps == 0 else f"⚠️ {open_gaps} Gap(s) Remaining"

    return {
        "recommendation": clean,
        "open_gaps": open_gaps,
        "status": status,
    }

def build_analysis_graph():
    g = StateGraph(AnalysisState)
    g.add_node("retrieve_rules",       retrieve_rules)
    g.add_node("analyse_gaps",         analyse_gaps)
    g.add_node("recommend_guideline",  recommend_guideline)

    g.set_entry_point("retrieve_rules")
    g.add_edge("retrieve_rules",      "analyse_gaps")
    g.add_edge("analyse_gaps",        "recommend_guideline")
    g.add_edge("recommend_guideline", END)
    return g.compile()

analysis_graph = build_analysis_graph()

# ============================================================================
# TABS
# ============================================================================
tab_kb, tab_app = st.tabs(["📚 Knowledge Base", "📝 Loan Applications"])

# ============================================================================
# TAB 1 — KNOWLEDGE BASE MANAGEMENT
# ============================================================================
with tab_kb:
    st.header("📚 Knowledge Base Management")
    st.markdown(f"""
    Manage the underwriting guideline documents stored in the **`{COLLECTION_RAW}`** Qdrant collection.
    The **`{COLLECTION_CURATED}`** collection is also queried during analysis (read-only).

    **Add** new guidelines, **Update** existing ones (re-upload with same name), or **Delete** outdated rules.
    """)

    # --- Current Documents ---
    st.subheader("📄 Current Documents in KB")
    registry = load_kb_registry()
    if registry:
        rows = []
        for name, info in registry.items():
            rows.append({
                "Document": name,
                "Guideline": info.get("guideline", "—"),
                "Chunks": info.get("chunk_count", 0),
                "Uploaded": info.get("uploaded_at", "—")[:19],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No documents in the knowledge base yet. Upload guideline PDFs below.")

    st.divider()

    # --- Upload / Update ---
    st.subheader("⬆️ Add or Update Guideline Documents")
    st.caption("If you upload a file with the same name as an existing document, it will **replace** the old version automatically.")

    col1, col2 = st.columns([3, 1])
    with col2:
        guideline_tag = st.selectbox("Guideline Type", ["Fannie Mae", "FHA", "General"], key="kb_guideline")
    with col1:
        kb_files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="kb_upload"
        )

    if kb_files and st.button("📤 Upload to Knowledge Base", type="primary", key="kb_go"):
        for f in kb_files:
            with st.spinner(f"Processing **{f.name}**..."):
                # If document already exists → delete old vectors first
                if f.name in registry:
                    old_ids = unregister_kb_document(f.name)
                    if old_ids:
                        try:
                            qdrant_client.delete(
                                collection_name=COLLECTION_RAW,
                                points_selector=PointIdsList(points=old_ids),
                            )
                            st.info(f"♻️ Replaced old version of **{f.name}** ({len(old_ids)} chunks removed)")
                        except Exception as e:
                            st.warning(f"Could not delete old vectors for {f.name}: {e}")

                # Extract, chunk, embed, upsert
                text = extract_text_from_pdf(f.read())
                docs = chunk_text(text, source=f.name, guideline=guideline_tag)
                ids  = [str(uuid.uuid4()) for _ in docs]
                vs_raw.add_documents(docs, ids=ids)

                # Register
                register_kb_document(f.name, guideline_tag, ids, len(docs))
                st.success(f"✅ **{f.name}** → {len(docs)} chunks stored ({guideline_tag})")

        st.rerun()

    st.divider()

    # --- Delete ---
    st.subheader("🗑️ Delete Documents from KB")
    registry = load_kb_registry()  # refresh
    if registry:
        to_delete = st.multiselect("Select documents to remove", list(registry.keys()), key="kb_delete_sel")
        if to_delete and st.button("🗑️ Delete Selected", type="secondary", key="kb_del_go"):
            for name in to_delete:
                old_ids = unregister_kb_document(name)
                if old_ids:
                    try:
                        qdrant_client.delete(
                            collection_name=COLLECTION_RAW,
                            points_selector=PointIdsList(points=old_ids),
                        )
                    except Exception:
                        pass
                st.success(f"🗑️ Deleted **{name}**")
            st.rerun()
    else:
        st.caption("Nothing to delete — the KB is empty.")

    st.divider()

    # --- Curated collection info ---
    st.subheader(f"📖 Curated Collection: `{COLLECTION_CURATED}`")
    if vs_curated is not None:
        try:
            info = qdrant_client.get_collection(COLLECTION_CURATED)
            st.metric("Curated Vectors", info.points_count)
            st.caption("This collection is **read-only** and queried automatically during gap analysis for high-signal rules.")
        except Exception as e:
            st.warning(f"Could not fetch curated collection info: {e}")
    else:
        st.info("Curated collection is not available. Only the raw Mortgage collection will be used.")


# ============================================================================
# TAB 2 — LOAN APPLICATIONS
# ============================================================================
with tab_app:
    st.header("📝 Loan Applications")

    # --- Session state init ---
    if "current_app" not in st.session_state:
        st.session_state.current_app = None

    # --- Open existing application ---
    with st.expander("📂 Open a Previous Application", expanded=False):
        apps = list_applications()
        if apps:
            st.dataframe(apps, use_container_width=True, hide_index=True)
            open_id = st.selectbox(
                "Select Application ID",
                [""] + [a["Application ID"] for a in apps],
                key="open_app_sel",
            )
            if open_id and st.button("📂 Open", key="open_app_go"):
                loaded = load_application(open_id)
                if loaded:
                    st.session_state.current_app = loaded
                    st.success(f"Opened **{open_id}**")
                    st.rerun()
                else:
                    st.error("Could not load application.")
        else:
            st.info("No previous applications found.")

    # --- Start new application ---
    if st.button("🆕 Start New Application", type="primary", key="new_app"):
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
        st.success(f"Created application **{app_id}**")
        st.rerun()

    st.divider()

    # --- Active application workspace ---
    app = st.session_state.current_app
    if app is None:
        st.info("Start a new application or open an existing one above.")
    else:
        # Header
        col_id, col_status, col_rec = st.columns(3)
        col_id.metric("Application ID", app["application_id"])
        col_status.metric("Status", app.get("status", "Draft"))
        col_rec.metric("Recommended", app.get("recommended_guideline", "—"))

        st.divider()

        # --- Upload evidence ---
        st.subheader("📄 Upload Evidence Documents")
        st.caption("Upload pay stubs, W-2s, tax returns, bank statements, gift letters, etc.")

        evidence_files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
            key=f"ev_upload_{app['application_id']}"
        )

        if evidence_files and st.button("📤 Add Evidence to Application", key="add_ev"):
            new_text = ""
            for f in evidence_files:
                with st.spinner(f"Extracting **{f.name}**..."):
                    text = extract_text_from_pdf(f.read())
                    new_text += f"\n\n--- {f.name} ---\n{text}"
                    if f.name not in app["evidence_files"]:
                        app["evidence_files"].append(f.name)
                    st.success(f"✅ Added **{f.name}**")

            app["evidence_text"] = app.get("evidence_text", "") + new_text
            save_application(app)
            st.rerun()

        # Show current evidence
        if app["evidence_files"]:
            st.markdown("**Evidence on file:** " + ", ".join(f"`{f}`" for f in app["evidence_files"]))
        else:
            st.info("No evidence uploaded yet.")

        st.divider()

        # --- Run gap analysis ---
        st.subheader("🔍 Run Gap Analysis")
        st.caption("Analyses your evidence against **both** Fannie Mae and FHA guidelines from the Knowledge Base (raw + curated).")

        if not app["evidence_files"]:
            st.warning("Upload at least one evidence document before running analysis.")
        elif not load_kb_registry() and vs_curated is None:
            st.warning("The Knowledge Base is empty and no curated collection is available. Upload guidelines in the KB tab first.")
        else:
            if st.button("🔍 Analyse Evidence", type="primary", key="run_analysis"):
                with st.spinner("Retrieving rules from both collections & analysing gaps..."):
                    result = analysis_graph.invoke({
                        "evidence_text":      app["evidence_text"],
                        "evidence_files":     app["evidence_files"],
                        "fnma_raw_rules":     "",
                        "fnma_curated_rules": "",
                        "fha_raw_rules":      "",
                        "fha_curated_rules":  "",
                        "gap_report":         "",
                        "recommendation":     "",
                        "open_gaps":          0,
                        "status":             "",
                    })

                # Save results
                run_record = {
                    "run_at": datetime.datetime.now().isoformat(),
                    "evidence_files": list(app["evidence_files"]),
                    "gap_report": result["gap_report"],
                    "recommendation": result["recommendation"],
                    "open_gaps": result["open_gaps"],
                    "status": result["status"],
                }
                app.setdefault("analysis_history", []).append(run_record)
                app["status"] = result["status"]
                app["open_gaps"] = result["open_gaps"]

                # Extract recommended guideline
                rec_text = result["recommendation"]
                if "Fannie Mae" in rec_text.split("\n")[0]:
                    app["recommended_guideline"] = "Fannie Mae"
                elif "FHA" in rec_text.split("\n")[0]:
                    app["recommended_guideline"] = "FHA"

                save_application(app)
                st.rerun()

        # --- Display latest analysis ---
        history = app.get("analysis_history", [])
        if history:
            latest = history[-1]

            st.divider()
            st.subheader("📊 Latest Gap Analysis")
            st.caption(f"Run at {latest['run_at'][:19]}  •  Evidence: {', '.join(latest['evidence_files'])}")
            st.markdown(latest["gap_report"])

            st.divider()
            st.subheader("🏆 Guideline Recommendation")
            st.markdown(latest["recommendation"])

            # --- History ---
            if len(history) > 1:
                st.divider()
                with st.expander(f"📜 Analysis History ({len(history)} runs)", expanded=False):
                    for i, h in enumerate(reversed(history), 1):
                        st.markdown(f"### Run {len(history) - i + 1} — {h['run_at'][:19]}")
                        st.markdown(f"**Status:** {h['status']}  •  **Gaps:** {h['open_gaps']}")
                        st.markdown(h["gap_report"][:500] + ("..." if len(h["gap_report"]) > 500 else ""))
                        st.divider()
