"""
MORTGAGE UNDERWRITING APP - LATEST V6 FINAL
Fixes:
1. NameError: load_kb_registry not defined
2. 413 TPM limit (8000) -> truncated prompts to 5000 tokens
3. 401 Invalid API Key -> fixed cache + gsk_ validation + auto clear
4. Loan apps disappearing -> stored in Qdrant collection loan_applications (persistent)
5. NEW: Re-upload handling - if you re-upload same file, it REPLACES old data and next gap analysis uses new data
"""

import streamlit as st
import os
import io
import json
import uuid
import datetime
import re
from pathlib import Path
from typing import TypedDict, List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList, PointStruct, PayloadSchemaType, VectorParams, Distance
from openai import OpenAI

st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Mortgage Underwriting System")
st.caption("Knowledge Base • Loan Apps in Qdrant (Persistent) • Re-upload supported")

# ---- KEYS ----
DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY", "").strip()

st.sidebar.header("🔑 API Configuration")
GROQ_KEY = st.sidebar.text_input("Groq API Key", value=DEFAULT_GROQ_KEY, type="password", help="gsk_... from console.groq.com/keys").strip()
QDRANT_URL = st.sidebar.text_input("Qdrant URL", value=DEFAULT_QDRANT_URL, help="https://xxx.qdrant.io").strip()
QDRANT_KEY = st.sidebar.text_input("Qdrant API Key", value=DEFAULT_QDRANT_KEY, type="password").strip()

if not GROQ_KEY or not QDRANT_URL or not QDRANT_KEY:
    st.sidebar.warning("Missing keys")
    st.error("Set GROQ_API_KEY (gsk_...), QDRANT_URL, QDRANT_KEY in HF Secrets + Factory Reboot")
    st.stop()
if not GROQ_KEY.startswith("gsk_"):
    st.error(f"Groq key should start with gsk_, got {GROQ_KEY[:5]}... Go to console.groq.com/keys")
    st.stop()

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
COLLECTION_RAW = "Mortgage"
COLLECTION_CURATED = "CuratedKnowledge_mortgage"
COLLECTION_APPS = "loan_applications"

APPLICATIONS_DIR = Path("loan_applications")
APPLICATIONS_DIR.mkdir(exist_ok=True)
KB_REGISTRY = Path("kb_registry.json")

# ---- KB REGISTRY ----
def rebuild_kb_registry_from_qdrant(client: QdrantClient, collection_name: str) -> dict:
    try:
        registry = {}
        offset = None
        while True:
            points, next_offset = client.scroll(collection_name=collection_name, limit=500, offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                payload = p.payload or {}
                meta = payload.get("metadata", {}) or payload
                source = str(meta.get("source") or payload.get("source") or "unknown.pdf")
                guideline = meta.get("guideline") or payload.get("guideline") or "Unknown"
                if source not in registry:
                    registry[source] = {"guideline": guideline, "point_ids": [], "chunk_count": 0, "uploaded_at": datetime.datetime.now().isoformat(), "rebuilt": True}
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
    if KB_REGISTRY.exists():
        try:
            with open(KB_REGISTRY) as f:
                data = json.load(f)
                if data:
                    return data
        except:
            pass
    if client is not None:
        return rebuild_kb_registry_from_qdrant(client, collection_name)
    return {}

def save_kb_registry(registry: dict):
    with open(KB_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2, default=str)

def register_kb_document(doc_name: str, guideline: str, point_ids: list[str], chunk_count: int):
    registry = load_kb_registry()
    registry[doc_name] = {"guideline": guideline, "point_ids": point_ids, "chunk_count": chunk_count, "uploaded_at": datetime.datetime.now().isoformat()}
    save_kb_registry(registry)

def unregister_kb_document(doc_name: str) -> list[str]:
    registry = load_kb_registry()
    entry = registry.pop(doc_name, None)
    save_kb_registry(registry)
    return entry["point_ids"] if entry else []

# ---- PDF HELPERS ----
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

def generate_app_id() -> str:
    return f"APP-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

# ---- QDRANT LOAN APPS PERSISTENCE ----
def ensure_applications_collection(client: QdrantClient):
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_APPS for c in collections)
        if not exists:
            client.create_collection(collection_name=COLLECTION_APPS, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
            client.create_payload_index(COLLECTION_APPS, field_name="app_id", field_schema=PayloadSchemaType.KEYWORD)
            st.sidebar.success(f"✅ Created {COLLECTION_APPS} collection")
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to create {COLLECTION_APPS}: {e}")
        return False

def _app_id_to_point_id(app_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, app_id))

def save_application(app_data: dict):
    """Save to Qdrant (persistent) + local backup"""
    try:
        qdrant_payload = dict(app_data)
        if len(qdrant_payload.get("evidence_text", "")) > 100000:
            qdrant_payload["evidence_text"] = qdrant_payload["evidence_text"][-100000:]
            qdrant_payload["_truncated"] = True
        point_id = _app_id_to_point_id(app_data["app_id"])
        dummy_vector = [0.0] * 384
        qdrant_client.upsert(collection_name=COLLECTION_APPS, points=[PointStruct(id=point_id, vector=dummy_vector, payload=qdrant_payload)])
    except Exception as e:
        st.warning(f"Qdrant save failed: {e}")
    try:
        path = APPLICATIONS_DIR / f"{app_data['app_id']}.json"
        with open(path, "w") as f:
            json.dump(app_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Local save failed: {e}")

def load_application(app_id: str) -> dict | None:
    try:
        point_id = _app_id_to_point_id(app_id)
        points = qdrant_client.retrieve(collection_name=COLLECTION_APPS, ids=[point_id], with_payload=True)
        if points and points[0].payload:
            return points[0].payload
    except Exception as e:
        print(f"Qdrant load failed: {e}")
    path = APPLICATIONS_DIR / f"{app_id}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except:
            pass
    return None

def list_applications() -> list[dict]:
    apps = []
    try:
        offset = None
        all_payloads = []
        while True:
            points, next_offset = qdrant_client.scroll(collection_name=COLLECTION_APPS, limit=100, offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                all_payloads.append(p.payload or {})
            if next_offset is None:
                break
            offset = next_offset
        all_payloads.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        for data in all_payloads:
            apps.append({"app_id": data.get("app_id", "Unknown"), "created": data.get("created_at", "Unknown"), "documents": len(data.get("evidence_files", [])), "analyses": len(data.get("analysis_history", [])), "status": data.get("status", "Open")})
        if apps:
            return apps
    except Exception as e:
        print(f"Qdrant list failed: {e}")
    for f in sorted(APPLICATIONS_DIR.glob("APP-*.json"), reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
                apps.append({"app_id": data.get("app_id", f.stem), "created": data.get("created_at", "Unknown"), "documents": len(data.get("evidence_files", [])), "analyses": len(data.get("analysis_history", [])), "status": data.get("status", "Open")})
        except:
            continue
    return apps

def delete_application(app_id: str):
    try:
        point_id = _app_id_to_point_id(app_id)
        qdrant_client.delete(collection_name=COLLECTION_APPS, points_selector=PointIdsList(points=[point_id]))
    except Exception as e:
        print(f"Qdrant delete failed: {e}")
    try:
        path = APPLICATIONS_DIR / f"{app_id}.json"
        if path.exists():
            path.unlink()
    except:
        pass

# ---- NEW: Remove old file content from evidence_text ----
def remove_file_from_evidence_text(evidence_text: str, filename: str) -> str:
    """Remove all sections belonging to filename from evidence_text"""
    # Pattern: --- filename (Page X) --- ... until next --- or end
    # We split by the marker
    pattern = rf"--- {re.escape(filename)} \(Page \d+\) ---\n.*?(?=\n--- |\Z)"
    # Use DOTALL to match across lines
    cleaned = re.sub(pattern, "", evidence_text, flags=re.DOTALL)
    # Also handle old format without page number
    pattern2 = rf"--- {re.escape(filename)} ---.*?(?=\n--- |\Z)"
    cleaned = re.sub(pattern2, "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ---- GROQ CLIENT ----
@st.cache_resource
def get_groq_client(api_key: str):
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

groq_client = get_groq_client(GROQ_KEY)

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    kwargs = {"model": GROQ_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": temperature, "max_tokens": 1500}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = groq_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        if "401" in str(e) or "invalid_api_key" in str(e).lower() or "invalid api key" in str(e).lower():
            st.cache_resource.clear()
            raise Exception(f"401 Invalid Groq API Key {GROQ_KEY[:10]}... Create new at console.groq.com/keys, set as HF Secret GROQ_API_KEY, Factory Reboot. {e}")
        raise

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client(q_url: str, q_key: str):
    return QdrantClient(url=q_url, api_key=q_key)

@st.cache_resource
def get_vector_store(q_url: str, q_key: str, collection_name: str):
    emb = get_embeddings()
    return QdrantVectorStore.from_existing_collection(embedding=emb, collection_name=collection_name, url=q_url, api_key=q_key)

def ensure_payload_indexes(client: QdrantClient, collection_name: str):
    for field in ("metadata.guideline", "metadata.source"):
        try:
            client.create_payload_index(collection_name=collection_name, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
        except:
            pass

try:
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_KEY)
    vs_raw = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_RAW)
    ensure_payload_indexes(qdrant_client, COLLECTION_RAW)
    ensure_applications_collection(qdrant_client)
    _startup_registry = load_kb_registry(client=qdrant_client, collection_name=COLLECTION_RAW)
    st.sidebar.success(f"✅ {COLLECTION_RAW} ({len(_startup_registry)} docs)")
    st.sidebar.success(f"✅ {COLLECTION_APPS} (persistent)")
except Exception as e:
    st.sidebar.error(f"❌ Qdrant connection failed: {e}")
    st.stop()

vs_curated = None
try:
    vs_curated = get_vector_store(QDRANT_URL, QDRANT_KEY, COLLECTION_CURATED)
    ensure_payload_indexes(qdrant_client, COLLECTION_CURATED)
    st.sidebar.success(f"✅ {COLLECTION_CURATED}")
except:
    st.sidebar.info(f"ℹ️ {COLLECTION_CURATED} not found")

class UnderwritingState(TypedDict):
    evidence_text: str
    evidence_files: List[str]
    fnma_rules: str
    fha_rules: str
    curated_rules: str
    gap_analysis: str
    recommendation: str

def retrieve_rules(state: UnderwritingState) -> dict:
    query = truncate_text(state["evidence_text"], 500)
    fnma_docs = []
    fha_docs = []
    try:
        fnma_docs = vs_raw.similarity_search(query, k=3, filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="Fannie Mae"))]))
    except:
        pass
    try:
        fha_docs = vs_raw.similarity_search(query, k=3, filter=Filter(must=[FieldCondition(key="metadata.guideline", match=MatchValue(value="FHA"))]))
    except:
        pass
    fnma_text = "\n\n".join(f"[{d.metadata.get('source','?')}]\n{truncate_text(d.page_content, 600)}" for d in fnma_docs)
    fha_text = "\n\n".join(f"[{d.metadata.get('source','?')}]\n{truncate_text(d.page_content, 600)}" for d in fha_docs)
    curated_text = ""
    if vs_curated:
        try:
            curated_docs = vs_curated.similarity_search(query, k=2)
            curated_text = "\n\n".join(truncate_text(d.page_content, 600) for d in curated_docs)
        except:
            pass
    return {"fnma_rules": truncate_text(fnma_text, 2000), "fha_rules": truncate_text(fha_text, 2000), "curated_rules": truncate_text(curated_text, 1000)}

def analyse_gaps(state: UnderwritingState) -> dict:
    system = "You are a senior mortgage underwriter. Compare evidence vs Fannie Mae/FHA. Be concise."
    user = f"""FILES: {', '.join(state['evidence_files'])}
EVIDENCE: {truncate_text(state['evidence_text'], 3000)}
FNMA: {truncate_text(state['fnma_rules'], 1500)}
FHA: {truncate_text(state['fha_rules'], 1500)}
CURATED: {truncate_text(state['curated_rules'], 800)}
Provide: 1. Evidence Summary 2. Fannie Mae Gaps 3. FHA Gaps 4. Critical Missing Docs"""
    return {"gap_analysis": call_llm(system, user, 0.2)}

def recommend_guideline(state: UnderwritingState) -> dict:
    system = "You are senior mortgage underwriter. Recommend Fannie Mae vs FHA concisely."
    user = f"""FILES: {state['evidence_files']}
GAPS: {truncate_text(state['gap_analysis'], 2000)}
FNMA: {truncate_text(state['fnma_rules'], 1000)}
FHA: {truncate_text(state['fha_rules'], 1000)}
Return: ### Recommendation: [Fannie Mae / FHA / Neither] **Confidence:** **Reasoning:** **Next Steps:** **Risk:**"""
    return {"recommendation": call_llm(system, user, 0.1)}

workflow = StateGraph(UnderwritingState)
workflow.add_node("retrieve_rules", retrieve_rules)
workflow.add_node("analyse_gaps", analyse_gaps)
workflow.add_node("recommend_guideline", recommend_guideline)
workflow.set_entry_point("retrieve_rules")
workflow.add_edge("retrieve_rules", "analyse_gaps")
workflow.add_edge("analyse_gaps", "recommend_guideline")
workflow.add_edge("recommend_guideline", END)
pipeline = workflow.compile()

tab1, tab2 = st.tabs(["📚 Knowledge Base", "📋 Loan Applications (Qdrant)"])

with tab1:
    st.subheader("📚 Guideline Knowledge Base")
    kb_registry = load_kb_registry()
    c1,c2,c3 = st.columns(3)
    c1.metric("Docs", len(kb_registry))
    c2.metric("Chunks", sum(v['chunk_count'] for v in kb_registry.values()) if kb_registry else 0)
    c3.metric("Types", len(set(v['guideline'] for v in kb_registry.values())) if kb_registry else 0)
    st.divider()
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
                    point_ids = vs_raw.add_documents(chunks)
                    str_ids = [str(x) for x in point_ids] if isinstance(point_ids, list) else [str(uuid.uuid4()) for _ in chunks]
                    register_kb_document(pdf_file.name, guideline_type, str_ids, len(chunks))
                    st.success(f"✅ {pdf_file.name} added ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Failed {pdf_file.name}: {e}")
    st.divider()
    st.subheader("Existing Docs")
    if kb_registry:
        for doc_name, info in kb_registry.items():
            c1,c2,c3,c4 = st.columns([3,1,1,1])
            c1.write(f"📄 {doc_name}")
            c2.write(f"{info['guideline']}")
            c3.write(f"{info['chunk_count']}")
            if c4.button("Delete", key=f"del_{doc_name}"):
                ids = unregister_kb_document(doc_name)
                if ids:
                    try:
                        qdrant_client.delete(collection_name=COLLECTION_RAW, points_selector=PointIdsList(points=ids))
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                st.rerun()

with tab2:
    st.subheader(f"📋 Loan Applications - Qdrant `{COLLECTION_APPS}` (Persistent + Re-upload)")
    st.caption("Re-upload same filename = replaces old data. Upload new filename = adds to existing. Gap analysis always uses latest combined evidence.")
    
    if "current_app" not in st.session_state:
        st.session_state.current_app = None
    
    col_new, col_reload = st.columns(2)
    if col_new.button("➕ New Application", use_container_width=True):
        app_id = generate_app_id()
        app_data = {"app_id": app_id, "created_at": datetime.datetime.now().isoformat(), "status": "Open", "evidence_files": [], "evidence_text": "", "analysis_history": []}
        save_application(app_data)
        st.session_state.current_app = app_data
        st.success(f"Created {app_id} in Qdrant")
        st.rerun()
    if col_reload.button("🔄 Refresh from Qdrant", use_container_width=True):
        st.rerun()
    
    apps = list_applications()
    st.write(f"Found {len(apps)} applications in Qdrant:")
    if apps:
        for app in apps[:20]:
            c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,1,1])
            c1.write(f"**{app['app_id']}**")
            c2.write(f"📅 {app['created'][:10]}")
            c3.write(f"📄 {app['documents']}")
            c4.write(f"🔍 {app['analyses']}")
            if c5.button("Open", key=f"open_{app['app_id']}"):
                st.session_state.current_app = load_application(app["app_id"])
                st.rerun()
            if c6.button("🗑️", key=f"del_app_{app['app_id']}"):
                delete_application(app["app_id"])
                if st.session_state.current_app and st.session_state.current_app["app_id"] == app["app_id"]:
                    st.session_state.current_app = None
                st.rerun()
    else:
        st.info("No apps in Qdrant yet. Create one above.")
    
    st.divider()
    app = st.session_state.current_app
    if app is None:
        st.info("👆 Start new app or open existing")
        st.stop()
    
    st.subheader(f"📋 {app['app_id']} (from Qdrant)")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Status", app["status"])
    col_s2.metric("Evidence Files", len(app["evidence_files"]))
    col_s3.metric("Analyses", len(app["analysis_history"]))
    col_s4.metric("Created", app["created_at"][:10])
    
    st.divider()
    st.subheader("📤 Upload Evidence Documents")
    st.info("💡 **Re-upload logic:** Same filename = replaces old content with new data. Different filename = adds to existing. Next Gap Analysis will use updated combined evidence.", icon="🔄")
    
    evidence_files = st.file_uploader("Upload evidence PDFs", type=["pdf"], accept_multiple_files=True, key=f"evidence_{app['app_id']}")
    if evidence_files:
        files_processed = []
        for ef in evidence_files:
            with st.spinner(f"Processing {ef.name}..."):
                # NEW LOGIC: If file already exists, remove old content first
                if ef.name in app["evidence_files"]:
                    st.warning(f"🔄 Re-upload detected: {ef.name} - replacing old data with new data")
                    app["evidence_text"] = remove_file_from_evidence_text(app["evidence_text"], ef.name)
                else:
                    app["evidence_files"].append(ef.name)
                
                docs = extract_pdf_text(ef.read(), source_name=ef.name)
                new_text = "\n\n".join(f"--- {ef.name} (Page {d.metadata['page']}) ---\n{d.page_content}" for d in docs)
                app["evidence_text"] += "\n\n" + new_text
                files_processed.append(ef.name)
        
        if files_processed:
            save_application(app)
            st.success(f"✅ {len(files_processed)} file(s) updated in Qdrant: {', '.join(files_processed)}. New total chars: {len(app['evidence_text'])}. Run Gap Analysis again to use new data.")
            st.rerun()
    
    if app["evidence_files"]:
        with st.expander(f"📂 Evidence Files ({len(app['evidence_files'])}) - stored in Qdrant"):
            for i, fname in enumerate(app["evidence_files"], 1):
                # Show char count for that file
                count = app["evidence_text"].count(f"--- {fname} (Page")
                st.write(f"{i}. 📄 {fname} ({count} pages extracted)")
            st.caption(f"Total evidence_text length: {len(app['evidence_text'])} chars - will be truncated to 3000 for analysis (to stay under TPM)")
    
    st.divider()
    st.subheader("🔍 Run Gap Analysis")
    kb_registry = load_kb_registry()
    if not kb_registry:
        st.error("KB empty")
    elif not app["evidence_files"]:
        st.warning("Upload evidence first")
    else:
        st.caption(f"Will analyze {len(app['evidence_files'])} files, {len(app['evidence_text'])} chars (truncated to 3000 for LLM)")
        if st.button("🚀 Run Gap Analysis (uses latest re-uploaded data)", use_container_width=True, type="primary"):
            with st.spinner("Analysing latest evidence..."):
                try:
                    result = pipeline.invoke({"evidence_text": app["evidence_text"], "evidence_files": app["evidence_files"], "fnma_rules": "", "fha_rules": "", "curated_rules": "", "gap_analysis": "", "recommendation": ""})
                    entry = {"timestamp": datetime.datetime.now().isoformat(), "evidence_count": len(app["evidence_files"]), "evidence_files": list(app["evidence_files"]), "gap_analysis": result["gap_analysis"], "recommendation": result["recommendation"], "evidence_text_length": len(app["evidence_text"])}
                    app["analysis_history"].append(entry)
                    save_application(app)
                    st.success("✅ Analysis complete and saved to Qdrant - includes your re-uploaded new data")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    if app["analysis_history"]:
        latest = app["analysis_history"][-1]
        st.divider()
        st.subheader(f"📊 Latest Results (Run #{len(app['analysis_history'])})")
        st.caption(f"{latest['timestamp'][:19].replace('T', ' at ')} • {latest['evidence_count']} files • {latest.get('evidence_text_length', 0)} chars analyzed")
        with st.container(border=True):
            st.markdown(latest["recommendation"])
        with st.container(border=True):
            st.markdown(latest["gap_analysis"])
        
        if len(app["analysis_history"]) > 1:
            with st.expander(f"📜 Previous Runs ({len(app['analysis_history'])-1} older)"):
                for entry in reversed(app["analysis_history"][:-1]):
                    st.markdown(f"**{entry['timestamp'][:19]}** - {entry['evidence_count']} files")
                    st.markdown(entry["recommendation"])
                    st.divider()
