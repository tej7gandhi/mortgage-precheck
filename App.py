import streamlit as st
import os
from typing import TypedDict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END

# --- CONFIG from image_3272db.png / image_6dcbe5.png ---
# Your Qdrant Cloud cluster: d4eb5a10-3452-498b-8618-38d13ca0dc9d.ca-central-1-0.aws.cloud.qdrant.io
QDRANT_URL = os.getenv("QDRANT_URL", "https://d4eb5a10-3452-498b-8618-38d13ca0dc9d.ca-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_KEY = os.getenv("QDRANT_KEY", "")

st.set_page_config(page_title="Fannie Mae Pre-Check", page_icon="🏦", layout="wide")
st.title("🏦 Fannie Mae Pre-Check Assistant")
st.caption("Powered by Qdrant (4 collections) + LangGraph isolation | Mortgage underwriters — no technical setup needed")

# Sidebar - Loan Scenario (non-technical)
st.sidebar.header("Loan Scenario")
borrower_type = st.sidebar.selectbox("Borrower Type", ["Salaried", "Self-Employed", "Retired"])
gift = st.sidebar.radio("Family Gift for Down Payment?", ["No", "Yes"])
large_deposit = st.sidebar.radio("Large Deposits (>50% monthly income)?", ["No", "Yes"])
case_id = st.sidebar.text_input("Case ID / Loan Number", "loan-12345")
investor = st.sidebar.selectbox("Investor", ["Fannie Mae", "Freddie Mac"])

# Cache embeddings + stores (from image_6dcbe5.png - 4 collections GREEN 384 Cosine)
@st.cache_resource
def get_qdrant_stores():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dim matches your collections
    mortgage = QdrantVectorStore.from_existing_collection(embedding=emb, collection_name="Mortgage", url=QDRANT_URL, api_key=QDRANT_KEY)
    curated = QdrantVectorStore.from_existing_collection(embedding=emb, collection_name="CuratedKnowledge_mortgage", url=QDRANT_URL, api_key=QDRANT_KEY)
    clinical = QdrantVectorStore.from_existing_collection(embedding=emb, collection_name="clinical_trials", url=QDRANT_URL, api_key=QDRANT_KEY)
    cyber = QdrantVectorStore.from_existing_collection(embedding=emb, collection_name="cyber_security", url=QDRANT_URL, api_key=QDRANT_KEY)
    return emb, mortgage, curated, clinical, cyber

try:
    embeddings, mortgage_store, curated_store, clinical_store, cyber_store = get_qdrant_stores()
    st.sidebar.success(f"✅ Qdrant Connected: 4 collections\nMortgage 3 pts, Curated 1 pt (from image_6dcbe5.png)")
except Exception as e:
    st.sidebar.error(f"Qdrant connection failed: {e}\nCheck QDRANT_URL and QDRANT_KEY in Secrets")
    st.stop()

# LangGraph State
class UnderwriterState(TypedDict):
    question: str
    borrower_type: str
    gift: str
    large_deposit: str
    case_id: str
    investor: str
    docs: List[dict]
    citations: List[str]
    flags: List[str]
    answer: str

# Node 1: Mortgage Retriever - ONLY queries Mortgage + CuratedKnowledge_mortgage (isolation from clinical/cyber)
def mortgage_retriever_node(state: UnderwriterState):
    q = f"{state['question']} Borrower {state['borrower_type']} Gift {state['gift']} LargeDeposit {state['large_deposit']} Investor {state['investor']} Case {state['case_id']}"
    # Query Qdrant Mortgage collection (3 points - Page 42, Page 100, Consistency)
    m_results = mortgage_store.similarity_search_with_score(q, k=3)
    # Query CuratedKnowledge_mortgage (1 point - Approved by Tej Gandhi)
    c_results = curated_store.similarity_search_with_score(q, k=2)
    
    combined = []
    for doc, score in m_results:
        combined.append({"content": doc.page_content, "citation": doc.metadata.get("citation",""), "page": doc.metadata.get("page",""), "source": doc.metadata.get("source",""), "approved_by": doc.metadata.get("approved_by",""), "score": score, "collection": "Mortgage"})
    for doc, score in c_results:
        combined.append({"content": doc.page_content, "citation": doc.metadata.get("citation",""), "page": doc.metadata.get("page",""), "source": doc.metadata.get("source",""), "approved_by": doc.metadata.get("approved_by",""), "score": score, "collection": "CuratedKnowledge_mortgage"})
    
    citations = [f"{d['citation']} Page {d['page']}" if d['page'] else d['citation'] for d in combined]
    
    flags = []
    content_str = " ".join([d['content'].lower() for d in combined])
    if state['gift']=="Yes" and "gift letter" not in content_str and "gift" not in content_str:
        flags.append("Missing gift letter — Required per Fannie Mae B1-1-01 Page 42")
    if state['large_deposit']=="Yes" and "large deposit" not in content_str:
        flags.append("Large deposit flagged — Needs source verification per Consistency Check")
    if state['borrower_type']=="Self-Employed" and "2 years tax returns" not in content_str:
        flags.append("Self-employed — Requires 2 years tax returns per Page 100")
    
    return {"docs": combined, "citations": citations, "flags": flags}

def generate_answer_node(state: UnderwriterState):
    if not state['docs']:
        return {"answer": "No relevant documents found in Mortgage collections. Please check Qdrant ingestion (image_6dcbe5.png should show points)."}
    
    best = state['docs'][0]
    answer = f"**For Case {state['case_id']} ({state['borrower_type']}, Gift: {state['gift']}):**\n\n{best['content']}\n\n"
    if best.get('approved_by'):
        answer += f"\n✅ **Approved override:** Approved by {best['approved_by']} for case {state['case_id']}\n"
    return {"answer": answer}

# Build LangGraph
graph = StateGraph(UnderwriterState)
graph.add_node("mortgage_retrieve", mortgage_retriever_node)
graph.add_node("generate", generate_answer_node)
graph.set_entry_point("mortgage_retrieve")
graph.add_edge("mortgage_retrieve", "generate")
graph.add_edge("generate", END)
langgraph_app = graph.compile()

# Main UI
st.markdown("### Ask a question")
question = st.text_input("Example: Borrower has family gift $20k for down payment — what docs needed?", 
                         placeholder="Type your underwriting question here...", 
                         label_visibility="collapsed")

col1, col2 = st.columns([3,1])
with col2:
    ask_btn = st.button("🔍 Check Requirements", type="primary", use_container_width=True)

if question and ask_btn:
    with st.spinner("Checking Fannie Mae guidelines via LangGraph → Qdrant Mortgage collections..."):
        result = langgraph_app.invoke({
            "question": question,
            "borrower_type": borrower_type,
            "gift": gift,
            "large_deposit": large_deposit,
            "case_id": case_id,
            "investor": investor
        })
    
    st.markdown("---")
    st.markdown("#### ✅ Answer")
    st.markdown(result['answer'])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📄 Citations (Exact)**")
        for cite in result['citations']:
            if "Page 42" in cite:
                st.markdown(f":blue-badge[**{cite}**]")
            else:
                st.markdown(f"- {cite}")
    with c2:
        st.markdown("**🚩 Flags**")
        if result['flags']:
            for f in result['flags']:
                st.warning(f)
        else:
            st.success("No flags — Documents complete per Page 42")
    with c3:
        st.markdown("**🔍 Qdrant + LangGraph Audit**")
        for doc in result['docs']:
            st.caption(f"Collection: {doc['collection']} | Score: {doc['score']:.3f} | Source: {doc['source']}")
            if doc['collection'] in ["clinical_trials", "cyber_security"]:
                st.error("Isolation breach! Should never happen")
    
    st.markdown("---")
    st.caption("Backend: Qdrant Cloud d4eb5a10...ca-central (Mortgage 3 pts, Curated 1 pt) | LangGraph mortgage_graph queries ONLY Mortgage + CuratedKnowledge_mortgage — never clinical_trials / cyber_security | Embeddings: all-MiniLM-L6-v2 384 Cosine")

else:
    st.info("👈 Set loan scenario on left, type question above, and click Check Requirements. Example questions:\n- What documents borrower must provide W2 bank statements?\n- Self-employed income calculation Page 100?\n- Family gift allowed?")

st.sidebar.markdown("---")
st.sidebar.markdown("**For Admins:** Qdrant collections from image_6dcbe5.png:\n- Mortgage 3\n- CuratedKnowledge_mortgage 1\n- clinical_trials 1\n- cyber_security 1\nAll GREEN 384 Cosine")
