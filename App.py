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
        with open(f) as fh:
            data = json.load(fh)
            apps.append({
                "app_id": data.get("app_id", f.stem),
                "created": data.get("created_at", "Unknown"),
                "documents": len(data.get("evidence_files", [])),
                "analyses": len(data.get("analysis_history", [])),
                "status": data.get("status", "Open"),
            })
    return apps

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
    """Retrieve relevant rules from both Qdrant collections for Fannie Mae & FHA."""
    query = state["evidence_text"][:1500]

    # --- Raw collection: filtered by guideline ---
    fnma_docs = vs_raw.similarity_search(
        query, k=8,
        filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="Fannie Mae"))]),
    )
    fha_docs = vs_raw.similarity_search(
        query, k=8,
        filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="FHA"))]),
    )

    fnma_text = "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in fnma_docs
    )
    fha_text = "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in fha_docs
    )

    # --- Curated collection: unfiltered (read-only, authoritative) ---
    curated_text = ""
    if vs_curated:
        curated_docs = vs_curated.similarity_search(query, k=10)
        curated_text = "\n\n".join(
            f"[Curated Rule | {d.metadata.get('source','?')}]\n{d.page_content}"
            for d in curated_docs
        )

    return {"fnma_rules": fnma_text, "fha_rules": fha_text, "curated_rules": curated_text}

def analyse_gaps(state: UnderwritingState) -> dict:
    """LLM analyses evidence against both guideline sets."""
    system = """You are an expert mortgage underwriter. Analyse the borrower's evidence against
BOTH Fannie Mae and FHA guidelines. Use the curated rules as authoritative references.

You MUST respond in **clean Markdown** using the EXACT structure below.
Use bullet points (- ) for lists. Use numbered lists (1. 2. 3.) for action items.
NEVER use <br> or HTML tags. NEVER output raw HTML.

---

## Fannie Mae Assessment

### ✅ Requirements Met
- Bullet each requirement that is fully satisfied
- Reference the specific guideline section

### ⚠️ Partially Met
- Bullet each requirement that has some but not enough evidence
- Explain what is present and what is still insufficient

### ❌ Gaps — Missing Evidence
1. **[Gap Name]** — Explain what document/data is missing and why it's required
2. **[Gap Name]** — Next gap
(Number each gap so the borrower can track them)

### 📄 Documents Still Needed
1. Specific document name — brief reason
2. Next document needed

### 📚 Key Guideline References
- Guideline section → what it requires

---

## FHA Assessment

### ✅ Requirements Met
- (same structure as above)

### ⚠️ Partially Met
- (same structure)

### ❌ Gaps — Missing Evidence
1. (same numbered structure)

### 📄 Documents Still Needed
1. (same structure)

### 📚 Key Guideline References
- (same structure)

---

Be specific. Cite guideline chapters and sections. If evidence is ambiguous, flag it under Partially Met."""

    user = f"""## Borrower Evidence
{state['evidence_text']}

## Evidence Files Uploaded
{', '.join(state['evidence_files'])}

## Fannie Mae Guidelines (from Knowledge Base)
{state['fnma_rules']}

## FHA Guidelines (from Knowledge Base)
{state['fha_rules']}

## Curated Authoritative Rules
{state['curated_rules']}"""

    result = call_llm(system, user, temperature=0.15)
    return {"gap_analysis": result}

def recommend_guideline(state: UnderwritingState) -> dict:
    """LLM recommends Fannie Mae or FHA based on the gap analysis."""
    system = """You are a senior mortgage advisor. Based on the gap analysis, recommend whether
Fannie Mae or FHA is the BETTER path to approval for this borrower.

Respond in **clean Markdown** only. No HTML tags. No <br> tags.

Use this EXACT structure:

---

## 🏆 Recommendation: [Fannie Mae / FHA]

### Why This Choice
A clear 2-3 sentence explanation of why this guideline is better for this borrower.

### Comparison Summary

| Factor | Fannie Mae | FHA |
|--------|-----------|-----|
| Outstanding Gaps | X gaps | Y gaps |
| Approval Difficulty | Easy/Medium/Hard | Easy/Medium/Hard |
| Down Payment | X% required | Y% required |
| Mortgage Insurance | PMI details | MIP details |
| DTI Flexibility | Stricter/More flexible | Stricter/More flexible |
| Overall Fit | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 📋 Priority Next Steps
1. **[Most important action]** — why this matters
2. **[Second action]** — brief reason
3. **[Third action]** — brief reason

### 🔄 When to Reconsider
Briefly state when the borrower should consider the other option instead.

---

Be decisive. Pick one. Don't hedge with "it depends"."""

    user = f"""## Gap Analysis Results
{state['gap_analysis']}

## Evidence Files
{', '.join(state['evidence_files'])}"""

    result = call_llm(system, user, temperature=0.2)
    return {"recommendation": result}

# Build the LangGraph pipeline
workflow = StateGraph(UnderwritingState)
workflow.add_node("retrieve_rules", retrieve_rules)
workflow.add_node("analyse_gaps", analyse_gaps)
workflow.add_node("recommend_guideline", recommend_guideline)
workflow.set_entry_point("retrieve_rules")
workflow.add_edge("retrieve_rules", "analyse_gaps")
workflow.add_edge("analyse_gaps", "recommend_guideline")
workflow.add_edge("recommend_guideline", END)
pipeline = workflow.compile()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — KNOWLEDGE BASE                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
tab_kb, tab_app = st.tabs(["📚 Knowledge Base", "📝 Loan Applications"])

with tab_kb:
    st.header("📚 Knowledge Base Management")
    st.info(
        "Documents uploaded here are **stored permanently** in Qdrant. "
        "You do **not** need to re-upload them each time — they persist across sessions "
        "and are automatically referenced for every loan application analysis.",
        icon="💡",
    )

    # --- Current Documents ---
    st.subheader("📂 Current Documents in Knowledge Base")
    registry = load_kb_registry()

    if registry:
        col_m1, col_m2, col_m3 = st.columns(3)
        fnma_count = sum(1 for v in registry.values() if v["guideline"] == "Fannie Mae")
        fha_count = sum(1 for v in registry.values() if v["guideline"] == "FHA")
        total_chunks = sum(v["chunk_count"] for v in registry.values())
        col_m1.metric("📄 Total Documents", len(registry))
        col_m2.metric("🏛️ Fannie Mae / 🏠 FHA", f"{fnma_count} / {fha_count}")
        col_m3.metric("🧩 Total Chunks", total_chunks)

        for doc_name, info in registry.items():
            with st.expander(f"{'🏛️' if info['guideline'] == 'Fannie Mae' else '🏠'} {doc_name}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Guideline:** {info['guideline']}")
                c2.write(f"**Chunks:** {info['chunk_count']}")
                c3.write(f"**Uploaded:** {info['uploaded_at'][:10]}")
                if st.button(f"🗑️ Delete", key=f"del_{doc_name}"):
                    point_ids = unregister_kb_document(doc_name)
                    if point_ids:
                        qdrant_client.delete(
                            collection_name=COLLECTION_RAW,
                            points_selector=PointIdsList(points=point_ids),
                        )
                    st.success(f"Deleted **{doc_name}** ({len(point_ids)} vectors removed)")
                    st.rerun()
    else:
        st.warning("No documents in the knowledge base yet. Upload guidelines below to get started.")

    # --- Curated Collection Info ---
    if vs_curated:
        curated_info = qdrant_client.get_collection(COLLECTION_CURATED)
        st.caption(f"📖 Curated collection `{COLLECTION_CURATED}` has **{curated_info.points_count}** vectors (read-only, auto-referenced)")

    st.divider()

    # --- Upload New / Update ---
    st.subheader("⬆️ Upload or Update Guidelines")
    with st.form("kb_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload guideline PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="If a file with the same name already exists, it will be **replaced** automatically."
        )
        guideline_type = st.selectbox("Guideline Type", ["Fannie Mae", "FHA"])
        submitted = st.form_submit_button("📤 Upload to Knowledge Base", use_container_width=True)

    if submitted and uploaded_files:
        for uf in uploaded_files:
            with st.status(f"Processing **{uf.name}**...", expanded=True) as status:
                # If document exists, delete old vectors first
                if uf.name in load_kb_registry():
                    st.write("🔄 Replacing existing document...")
                    old_ids = unregister_kb_document(uf.name)
                    if old_ids:
                        qdrant_client.delete(
                            collection_name=COLLECTION_RAW,
                            points_selector=PointIdsList(points=old_ids),
                        )
                    st.write(f"   Removed {len(old_ids)} old vectors")

                # Extract & chunk
                st.write("📄 Extracting text from PDF...")
                raw_docs = extract_pdf_text(uf.read(), source_name=uf.name)
                st.write(f"   Found {len(raw_docs)} pages")

                st.write("✂️ Chunking documents...")
                chunks = chunk_documents(raw_docs)
                for c in chunks:
                    c.metadata["guideline"] = guideline_type
                    c.metadata["citation"] = f"{uf.name} (p.{c.metadata.get('page', '?')})"
                st.write(f"   Created {len(chunks)} chunks")

                # Generate deterministic IDs
                point_ids = []
                for i, c in enumerate(chunks):
                    raw_id = f"{uf.name}_{i}_{hashlib.md5(c.page_content.encode()).hexdigest()}"
                    point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id)))

                # Upload to Qdrant
                st.write("⬆️ Uploading to Qdrant...")
                vs_raw.add_documents(chunks, ids=point_ids)

                # Register
                register_kb_document(uf.name, guideline_type, point_ids, len(chunks))
                status.update(label=f"✅ **{uf.name}** — {len(chunks)} chunks uploaded", state="complete")
        st.rerun()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — LOAN APPLICATIONS                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
with tab_app:
    st.header("📝 Loan Applications")

    # --- Session state init ---
    if "current_app" not in st.session_state:
        st.session_state.current_app = None

    # --- Open existing or start new ---
    col_new, col_open = st.columns([1, 2])
    with col_new:
        if st.button("🆕 Start New Application", use_container_width=True, type="primary"):
            new_id = generate_app_id()
            st.session_state.current_app = {
                "app_id": new_id,
                "created_at": datetime.datetime.now().isoformat(),
                "evidence_files": [],
                "evidence_text": "",
                "analysis_history": [],
                "status": "Open",
            }
            save_application(st.session_state.current_app)
            st.rerun()

    with col_open:
        apps = list_applications()
        if apps:
            with st.expander("📂 Open a Previous Application"):
                for app in apps:
                    status_icon = "🟢" if app["status"] == "Open" else "✅"
                    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
                    c1.write(f"**{app['app_id']}**")
                    c2.write(f"📅 {app['created'][:10]}")
                    c3.write(f"📄 {app['documents']} docs")
                    c4.write(f"🔍 {app['analyses']} runs")
                    if c5.button("Open", key=f"open_{app['app_id']}"):
                        st.session_state.current_app = load_application(app["app_id"])
                        st.rerun()

    st.divider()

    # --- Active Application ---
    app = st.session_state.current_app

    if app is None:
        st.info("👆 Start a new application or open an existing one to begin.", icon="📋")
        st.stop()

    # --- Application Header ---
    st.subheader(f"📋 Application: {app['app_id']}")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Status", app["status"])
    col_s2.metric("Evidence Files", len(app["evidence_files"]))
    col_s3.metric("Analyses Run", len(app["analysis_history"]))
    col_s4.metric("Created", app["created_at"][:10])

    st.divider()

    # --- Upload Evidence ---
    st.subheader("📤 Upload Evidence Documents")
    st.caption("Upload borrower documents: W-2s, tax returns, bank statements, pay stubs, gift letters, appraisals, etc.")

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
            st.success(f"✅ {len(evidence_files)} file(s) processed and added to the application.")

    # Show current evidence
    if app["evidence_files"]:
        with st.expander(f"📂 Evidence Files ({len(app['evidence_files'])})"):
            for i, fname in enumerate(app["evidence_files"], 1):
                st.write(f"{i}. 📄 {fname}")
    else:
        st.warning("No evidence files uploaded yet.")

    st.divider()

    # --- Run Analysis ---
    st.subheader("🔍 Run Gap Analysis")
    kb_registry = load_kb_registry()
    if not kb_registry:
        st.error("⚠️ Knowledge Base is empty. Please upload guidelines in the **Knowledge Base** tab first.")
    elif not app["evidence_files"]:
        st.warning("⚠️ Upload at least one evidence document before running the analysis.")
    else:
        fnma_in_kb = any(v["guideline"] == "Fannie Mae" for v in kb_registry.values())
        fha_in_kb = any(v["guideline"] == "FHA" for v in kb_registry.values())
        st.caption(
            f"Knowledge Base: {'🏛️ Fannie Mae ✅' if fnma_in_kb else '🏛️ Fannie Mae ❌'}"
            f"  |  {'🏠 FHA ✅' if fha_in_kb else '🏠 FHA ❌'}"
            f"  |  📄 {len(kb_registry)} documents  |  🧩 {sum(v['chunk_count'] for v in kb_registry.values())} chunks"
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

                    # Save analysis to history
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

    # --- Display Latest Results ---
    if app["analysis_history"]:
        latest = app["analysis_history"][-1]

        st.divider()
        st.subheader("📊 Latest Analysis Results")
        st.caption(f"Run #{len(app['analysis_history'])}  •  {latest['timestamp'][:19].replace('T', ' at ')}  •  Based on {latest['evidence_count']} document(s)")

        # --- Recommendation (show first — it's the most important) ---
        with st.container(border=True):
            st.markdown(latest["recommendation"])

        st.divider()

        # --- Detailed Gap Analysis ---
        with st.container(border=True):
            st.markdown("## 📋 Detailed Gap Analysis")
            st.markdown(latest["gap_analysis"])

        # --- Next Steps ---
        st.divider()
        st.info(
            "**📤 Upload additional documents above** to cover the identified gaps, "
            "then click **Run Gap Analysis** again. The analysis will update to reflect "
            "the newly submitted evidence.",
            icon="💡"
        )

        # --- Analysis History ---
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
