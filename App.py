"""
MORTGAGE UNDERWRITING APP - V11 FINAL - Button text = "Run Gap Assessment"
All fixes: Qdrant persistence, supersede logic, clear all evidence, no refresh bug, 401, 413, NameError
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Mortgage Underwriting System")

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY", "").strip()

st.sidebar.header("🔑 API Configuration")
GROQ_KEY = st.sidebar.text_input("Groq API Key", value=DEFAULT_GROQ_KEY, type="password").strip()
QDRANT_URL = st.sidebar.text_input("Qdrant URL", value=DEFAULT_QDRANT_URL).strip()
QDRANT_KEY = st.sidebar.text_input("Qdrant API Key", value=DEFAULT_QDRANT_KEY, type="password").strip()

if not GROQ_KEY or not QDRANT_URL or not QDRANT_KEY:
    st.error("Set GROQ_API_KEY, QDRANT_URL, QDRANT_KEY in Secrets")
    st.stop()
if not GROQ_KEY.startswith("gsk_"):
    st.error(f"Invalid Groq key")
    st.stop()

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
COLLECTION_RAW = "Mortgage"
COLLECTION_CURATED = "CuratedKnowledge_mortgage"
COLLECTION_APPS = "loan_applications"
APPLICATIONS_DIR = Path("loan_applications")
APPLICATIONS_DIR.mkdir(exist_ok=True)
KB_REGISTRY = Path("kb_registry.json")

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
    except:
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

def ensure_applications_collection(client: QdrantClient):
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_APPS for c in collections)
        if not exists:
            client.create_collection(collection_name=COLLECTION_APPS, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
            client.create_payload_index(COLLECTION_APPS, field_name="app_id", field_schema=PayloadSchemaType.KEYWORD)
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to create {COLLECTION_APPS}: {e}")
        return False

def _app_id_to_point_id(app_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, app_id))

def save_application(app_data: dict):
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
    except:
        pass

def load_application(app_id: str) -> dict | None:
    try:
        point_id = _app_id_to_point_id(app_id)
        points = qdrant_client.retrieve(collection_name=COLLECTION_APPS, ids=[point_id], with_payload=True)
        if points and points[0].payload:
            return points[0].payload
    except:
        pass
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
    except:
        pass
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
    except:
        pass
    try:
        path = APPLICATIONS_DIR / f"{app_id}.json"
        if path.exists():
            path.unlink()
    except:
        pass

def remove_file_from_evidence_text(evidence_text: str, filename: str) -> str:
    pattern = rf"--- {re.escape(filename)} \(Page \d+\) ---\n.*?(?=\n--- |\Z)"
    cleaned = re.sub(pattern, "", evidence_text, flags=re.DOTALL)
    pattern2 = rf"--- {re.escape(filename)} ---.*?(?=\n--- |\Z)"
    cleaned = re.sub(pattern2, "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

@st.cache_resource
def get_groq_client(api_key: str):
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

groq_client = get_groq_client(GROQ_KEY)

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    kwargs = {"model": GROQ_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": temperature, "max_tokens": 1800}
    try:
        response = groq_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        if "401" in str(e) or "invalid_api_key" in str(e).lower():
            st.cache_resource.clear()
            raise Exception(f"401 Invalid Groq API Key {GROQ_KEY[:10]}... Create new at console.groq.com/keys. {e}")
        raise

@st.cache_resource
def get_embeddings():
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client(q_url: str, q_key: str):
    return QdrantClient(url=q_url, api_key=q_key, check_compatibility=False)

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
except:
    pass

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
    system = """You are a senior mortgage underwriter. 
CRITICAL RULE: Evidence may contain old failing values and new corrected values.
If you see phrases like 'SUPERSEDES ALL PRIOR', 'CURRENT STATUS', 'REPLACES', 'USE THIS VALUE', or 'THIS DOCUMENT REPLACES ALL PRIOR EVIDENCE' - those values OVERRIDE all older values.
Always use the MOST RECENT and EXPLICIT 'CURRENT STATUS' or 'NEW CORRECTED VALUE' when there is a conflict.
If a document says 'Previous 580 SUPERSEDED - New score is 750', use 750, NOT 580."""
    user = f"""FILES: {', '.join(state['evidence_files'])}
EVIDENCE (use CURRENT STATUS values):
{truncate_text(state['evidence_text'], 4000)}
FNMA RULES: {truncate_text(state['fnma_rules'], 1500)}
FHA RULES: {truncate_text(state['fha_rules'], 1500)}
CURATED: {truncate_text(state['curated_rules'], 800)}
Provide: 1. Evidence Summary using CURRENT STATUS values only 2. Fannie Mae Gaps 3. FHA Gaps 4. Critical Missing Docs"""
    return {"gap_analysis": call_llm(system, user, 0.2)}

def recommend_guideline(state: UnderwritingState) -> dict:
    system = """You are senior mortgage underwriter. Recommend Fannie Mae vs FHA.
CRITICAL: Use CURRENT STATUS values if evidence contains SUPERSEDED markers."""
    user = f"""FILES: {state['evidence_files']}
GAPS: {truncate_text(state['gap_analysis'], 2500)}
FNMA: {truncate_text(state['fnma_rules'], 1000)}
FHA: {truncate_text(state['fha_rules'], 1000)}
Return: ### Recommendation: [Fannie Mae / FHA / Both / Neither] **Confidence:** **Reasoning:** **Next Steps:** **Risk:**"""
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
    st.subheader("📚 Knowledge Base")
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

with tab2:
    st.subheader(f"📋 Loan Applications - Qdrant `{COLLECTION_APPS}`")
    
    if "current_app" not in st.session_state:
        st.session_state.current_app = None
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0

    col_new, col_reload = st.columns(2)
    if col_new.button("➕ New Application (Clean - No Old Docs)", use_container_width=True, type="primary"):
        app_id = generate_app_id()
        app_data = {"app_id": app_id, "created_at": datetime.datetime.now().isoformat(), "status": "Open", "evidence_files": [], "evidence_text": "", "analysis_history": []}
        save_application(app_data)
        st.session_state.current_app = app_data
        st.session_state.uploader_nonce += 1
        st.success(f"Created clean app {app_id}")
        st.rerun()
    if col_reload.button("🔄 Refresh from Qdrant", use_container_width=True):
        st.rerun()
    
    apps = list_applications()
    if apps:
        for app in apps[:20]:
            c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,1,1])
            c1.write(f"**{app['app_id']}**")
            c2.write(f"📅 {app['created'][:10]}")
            c3.write(f"📄 {app['documents']}")
            c4.write(f"🔍 {app['analyses']}")
            if c5.button("Open", key=f"open_{app['app_id']}"):
                st.session_state.current_app = load_application(app["app_id"])
                st.session_state.uploader_nonce += 1
                st.rerun()
            if c6.button("🗑️", key=f"del_app_{app['app_id']}"):
                delete_application(app["app_id"])
                if st.session_state.current_app and st.session_state.current_app["app_id"] == app["app_id"]:
                    st.session_state.current_app = None
                st.rerun()
    
    st.divider()
    app = st.session_state.current_app
    if app is None:
        st.info("👆 Create New Application or open existing")
        st.stop()
    
    st.subheader(f"📋 {app['app_id']} (from Qdrant)")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Status", app["status"])
    col_s2.metric("Evidence Files", len(app["evidence_files"]))
    col_s3.metric("Analyses", len(app["analysis_history"]))
    col_s4.metric("Created", app["created_at"][:10])
    
    st.divider()
    st.subheader("📤 Upload Evidence Documents")
    
    col_clear, col_info = st.columns([1,2])
    with col_clear:
        if st.button("🗑️ Clear ALL Old Evidence", key=f"clear_{app['app_id']}", type="secondary"):
            app["evidence_files"] = []
            app["evidence_text"] = ""
            save_application(app)
            st.session_state.current_app = app
            st.success("✅ All old evidence cleared from Qdrant")
            st.rerun()
    with col_info:
        st.info("💡 Clear old evidence then upload only FINAL_ALL_GAPS_COVERED.pdf to test supersede", icon="🔄")
    
    with st.form(key=f"upload_form_{app['app_id']}_{st.session_state.uploader_nonce}", clear_on_submit=True):
        uploaded_files = st.file_uploader("Select PDFs - Same name replaces, new name adds", type=["pdf"], accept_multiple_files=True)
        submit_upload = st.form_submit_button("📤 Upload & Save to Qdrant", use_container_width=True, type="primary")
        
        if submit_upload and uploaded_files:
            files_processed = []
            for ef in uploaded_files:
                if ef.name in app["evidence_files"]:
                    app["evidence_text"] = remove_file_from_evidence_text(app["evidence_text"], ef.name)
                else:
                    app["evidence_files"].append(ef.name)
                
                docs = extract_pdf_text(ef.read(), source_name=ef.name)
                new_text = "\n\n".join(f"--- {ef.name} (Page {d.metadata['page']}) ---\n{d.page_content}" for d in docs)
                app["evidence_text"] += "\n\n" + new_text
                files_processed.append(ef.name)
            
            if files_processed:
                save_application(app)
                st.session_state.current_app = app
                st.success(f"✅ Saved {len(files_processed)} file(s) to Qdrant: {', '.join(files_processed)}")
                st.session_state.uploader_nonce += 1
    
    if app["evidence_files"]:
        st.subheader(f"📂 Evidence Files ({len(app['evidence_files'])})")
        for i, fname in enumerate(app["evidence_files"], 1):
            c1, c2 = st.columns([3,1])
            c1.write(f"{i}. 📄 {fname}")
            if c2.button(f"❌ Delete", key=f"del_file_{app['app_id']}_{fname}_{i}"):
                app["evidence_text"] = remove_file_from_evidence_text(app["evidence_text"], fname)
                if fname in app["evidence_files"]:
                    app["evidence_files"].remove(fname)
                save_application(app)
                st.session_state.current_app = app
                st.success(f"Deleted {fname} from Qdrant")
                st.rerun()
    
    st.divider()
    st.subheader("🔍 Gap Assessment")
    kb_registry = load_kb_registry()
    if not kb_registry:
        st.error("KB empty - upload guidelines first")
    elif not app["evidence_files"]:
        st.warning("Upload evidence first")
    else:
        # EXACT TEXT REQUESTED BY USER
        if st.button("Run Gap Assessment", use_container_width=True, type="primary", key=f"gap_btn_{app['app_id']}_{len(app['evidence_files'])}_{len(app['analysis_history'])}"):
            with st.spinner("Analysing... 20-30 sec..."):
                try:
                    result = pipeline.invoke({"evidence_text": app["evidence_text"], "evidence_files": app["evidence_files"], "fnma_rules": "", "fha_rules": "", "curated_rules": "", "gap_analysis": "", "recommendation": ""})
                    entry = {"timestamp": datetime.datetime.now().isoformat(), "evidence_count": len(app["evidence_files"]), "evidence_files": list(app["evidence_files"]), "gap_analysis": result["gap_analysis"], "recommendation": result["recommendation"], "evidence_text_length": len(app["evidence_text"])}
                    app["analysis_history"].append(entry)
                    save_application(app)
                    st.session_state.current_app = app
                    st.success("✅ Assessment complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    if app["analysis_history"]:
        latest = app["analysis_history"][-1]
        st.divider()
        st.subheader(f"📊 Latest Results (Run #{len(app['analysis_history'])})")
        st.caption(f"{latest['timestamp'][:19].replace('T', ' at ')} • {latest['evidence_count']} files")
        with st.container(border=True):
            st.markdown(latest["recommendation"])
        with st.container(border=True):
            st.markdown("## 📋 Detailed Gap Analysis")
            st.markdown(latest["gap_analysis"])
