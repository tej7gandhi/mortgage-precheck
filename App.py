"""
MORTGAGE UNDERWRITING APP
Assess loans against Fannie Mae & FHA guidelines simultaneously
"""

import streamlit as st
import os
from typing import TypedDict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END

# ============================================================================
# STEP 1: CONFIGURATION - API KEYS & DATABASE CONNECTION
# ============================================================================

st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Dual Guideline Mortgage Underwriter")
st.caption("Fannie Mae + FHA Assessment")

# Read default API keys from environment (or leave blank for user input)
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "")
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY", "")

# ============================================================================
# STEP 2: SIDEBAR - USER INPUTS FOR LOAN SCENARIO
# ============================================================================

st.sidebar.header("📋 Loan Details")

# Borrower info
borrower_type = st.sidebar.selectbox(
    "Borrower Type",
    ["Salaried", "Self-Employed", "Retired"]
)

# Down payment info
gift = st.sidebar.radio(
    "Family Gift for Down Payment?",
    ["No", "Yes"]
)

# Deposit info
large_deposit = st.sidebar.radio(
    "Large Deposits (>50% monthly income)?",
    ["No", "Yes"]
)

# Case identifier
case_id = st.sidebar.text_input(
    "Case ID / Loan Number",
    "loan-12345"
)

# ============================================================================
# STEP 3: SIDEBAR - CHOOSE WHICH GUIDELINES TO ASSESS
# ============================================================================

st.sidebar.subheader("✅ Which Guidelines?")
assess_fannie_mae = st.sidebar.checkbox("Fannie Mae", value=True)
assess_fha = st.sidebar.checkbox("FHA", value=True)

# ============================================================================
# STEP 4: SIDEBAR - API CONFIGURATION (User can plug in their keys)
# ============================================================================

st.sidebar.subheader("🔑 Qdrant API Config")

qdrant_url = st.sidebar.text_input(
    "Qdrant URL",
    value=DEFAULT_QDRANT_URL,
    type="default"
)

qdrant_key = st.sidebar.text_input(
    "Qdrant API Key",
    value=DEFAULT_QDRANT_KEY,
    type="password"
)

# Use sidebar inputs, fall back to defaults if empty
QDRANT_URL = qdrant_url if qdrant_url else DEFAULT_QDRANT_URL
QDRANT_KEY = qdrant_key if qdrant_key else DEFAULT_QDRANT_KEY

# ============================================================================
# STEP 5: CONNECT TO QDRANT VECTOR DATABASE
# ============================================================================

@st.cache_resource
def connect_to_qdrant():
    """
    Connect to Qdrant and load embedding collections.
    This runs once and is cached for speed.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load mortgage guidelines & knowledge
    mortgage_db = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="Mortgage",
        url=QDRANT_URL,
        api_key=QDRANT_KEY
    )
    
    curated_db = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="CuratedKnowledge_mortgage",
        url=QDRANT_URL,
        api_key=QDRANT_KEY
    )
    
    return embeddings, mortgage_db, curated_db

# Try to connect; show error if it fails
try:
    embeddings, mortgage_db, curated_db = connect_to_qdrant()
    st.sidebar.success("✅ Connected to Qdrant")
except Exception as e:
    st.sidebar.error(f"❌ Qdrant Connection Failed:\n{e}")
    st.stop()

# ============================================================================
# STEP 6: DEFINE THE STATE MACHINE (What data flows through the workflow)
# ============================================================================

class LoanState(TypedDict):
    """
    This is the data that flows through the underwriting workflow.
    """
    question: str                 # User's question
    borrower_type: str            # "Salaried", "Self-Employed", "Retired"
    gift: str                     # "Yes" or "No"
    large_deposit: str            # "Yes" or "No"
    case_id: str                  # Loan number
    guideline: str                # "Fannie Mae" or "FHA"
    retrieved_docs: List[dict]    # Documents found from Qdrant
    citations: List[str]          # Source citations
    compliance_flags: List[str]   # Red flags / compliance issues
    answer: str                   # Final underwriting decision

# ============================================================================
# STEP 7: FANNIE MAE COMPLIANCE RULES
# ============================================================================

def check_fannie_mae_rules(state: LoanState, retrieved_content: str) -> List[str]:
    """
    Check if loan meets Fannie Mae requirements.
    Returns list of compliance issues (if any).
    """
    flags = []
    
    # RULE 1: Gift funds require gift letter
    if state['gift'] == "Yes":
        if "gift letter" not in retrieved_content.lower():
            flags.append("❌ FANNIE MAE: Missing gift letter (Required per B1-1-01 Page 42)")
    
    # RULE 2: Large deposits need source verification
    if state['large_deposit'] == "Yes":
        if "large deposit" not in retrieved_content.lower():
            flags.append("❌ FANNIE MAE: Large deposit needs source verification (Consistency Check)")
    
    # RULE 3: Self-employed = 2 years of tax returns
    if state['borrower_type'] == "Self-Employed":
        if "2 years tax returns" not in retrieved_content.lower():
            flags.append("❌ FANNIE MAE: Self-employed requires 2 years tax returns (Page 100)")
    
    # RULE 4: Retired = document retirement income
    if state['borrower_type'] == "Retired":
        if "retirement account" not in retrieved_content.lower():
            flags.append("❌ FANNIE MAE: Retired borrower must document retirement income")
    
    return flags

# ============================================================================
# STEP 8: FHA COMPLIANCE RULES
# ============================================================================

def check_fha_rules(state: LoanState, retrieved_content: str) -> List[str]:
    """
    Check if loan meets FHA requirements.
    Returns list of compliance issues (if any).
    """
    flags = []
    
    # RULE 1: Gift funds require gift letter
    if state['gift'] == "Yes":
        if "gift letter" not in retrieved_content.lower():
            flags.append("❌ FHA: Missing gift letter (Required per 4155.1 Section 2-1)")
    
    # RULE 2: Self-employed = audited tax returns
    if state['borrower_type'] == "Self-Employed":
        if "2 years tax returns" not in retrieved_content.lower():
            flags.append("❌ FHA: Self-employed requires 2 years audited tax returns (4155.1)")
    
    # RULE 3: Large cash deposits = 2 months seasoning proof
    if state['large_deposit'] == "Yes":
        if "cash deposit" not in retrieved_content.lower():
            flags.append("❌ FHA: Large cash deposit needs 2 months seasoning verification")
    
    # RULE 4: Retired = verify Social Security income
    if state['borrower_type'] == "Retired":
        if "social security" not in retrieved_content.lower():
            flags.append("❌ FHA: Retired borrower must verify Social Security income")
    
    return flags

# ============================================================================
# STEP 9: WORKFLOW NODE 1 - RETRIEVE RELEVANT DOCUMENTS
# ============================================================================

def retrieve_documents(state: LoanState):
    """
    Search Qdrant for relevant mortgage guidelines.
    Builds query from: guideline type + borrower info + question
    """
    
    # Build a smart search query
    guideline_prefix = f"[{state['guideline']}]" if state.get('guideline') else ""
    search_query = (
        f"{guideline_prefix} {state['question']} "
        f"Borrower {state['borrower_type']} "
        f"Gift {state['gift']} "
        f"LargeDeposit {state['large_deposit']} "
        f"Case {state['case_id']}"
    )
    
    # Search both knowledge bases
    mortgage_results = mortgage_db.similarity_search_with_score(search_query, k=3)
    curated_results = curated_db.similarity_search_with_score(search_query, k=2)
    
    # Combine results
    combined_docs = []
    
    for doc, score in mortgage_results:
        combined_docs.append({
            "content": doc.page_content,
            "citation": doc.metadata.get("citation", "Unknown"),
            "page": doc.metadata.get("page", ""),
            "source": doc.metadata.get("source", ""),
            "approved_by": doc.metadata.get("approved_by", ""),
            "score": score,
            "collection": "Mortgage"
        })
    
    for doc, score in curated_results:
        combined_docs.append({
            "content": doc.page_content,
            "citation": doc.metadata.get("citation", "Unknown"),
            "page": doc.metadata.get("page", ""),
            "source": doc.metadata.get("source", ""),
            "approved_by": doc.metadata.get("approved_by", ""),
            "score": score,
            "collection": "Curated Knowledge"
        })
    
    # Build citations
    citations = [
        f"{d['citation']} (Page {d['page']})" if d['page'] else d['citation']
        for d in combined_docs
    ]
    
    # Check compliance rules
    content_str = " ".join([d['content'].lower() for d in combined_docs])
    
    if state.get('guideline') == "Fannie Mae":
        compliance_flags = check_fannie_mae_rules(state, content_str)
    elif state.get('guideline') == "FHA":
        compliance_flags = check_fha_rules(state, content_str)
    else:
        compliance_flags = []
    
    return {
        "retrieved_docs": combined_docs,
        "citations": citations,
        "compliance_flags": compliance_flags
    }

# ============================================================================
# STEP 10: WORKFLOW NODE 2 - GENERATE UNDERWRITING DECISION
# ============================================================================

def generate_decision(state: LoanState):
    """
    Create the final underwriting assessment.
    Includes guideline info, compliance flags, and recommendation.
    """
    
    if not state['retrieved_docs']:
        answer = f"⚠️ No relevant documents found for {state.get('guideline', 'selected')} guidelines."
    else:
        best_doc = state['retrieved_docs'][0]  # Best match
        
        answer = (
            f"📋 **{state.get('guideline', 'Guideline')} Assessment**\n"
            f"Case: {state['case_id']} | Borrower: {state['borrower_type']} | Gift: {state['gift']}\n\n"
            f"**Guideline Guidance:**\n{best_doc['content']}\n\n"
        )
        
        # Add compliance flags
        if state['compliance_flags']:
            answer += "**⚠️ Compliance Issues:**\n"
            for flag in state['compliance_flags']:
                answer += f"{flag}\n"
        else:
            answer += "**✅ All compliance checks passed**\n"
        
        # Add citations
        if state['citations']:
            answer += f"\n**Sources:** {', '.join(state['citations'][:2])}"
    
    return {"answer": answer}

# ============================================================================
# STEP 11: BUILD THE WORKFLOW (LangGraph)
# ============================================================================

workflow = StateGraph(LoanState)

# Add the two steps
workflow.add_node("retrieve_docs", retrieve_documents)
workflow.add_node("generate_answer", generate_decision)

# Define the flow: retrieve → generate → done
workflow.set_entry_point("retrieve_docs")
workflow.add_edge("retrieve_docs", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile into executable app
underwriter_app = workflow.compile()

# ============================================================================
# STEP 12: MAIN UI - UPLOAD DOCUMENTS
# ============================================================================

st.markdown("### 📄 Upload Loan Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs, Word docs, or text files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
    for file in uploaded_files:
        st.write(f"  • {file.name}")

# ============================================================================
# STEP 13: MAIN UI - ASK QUESTION
# ============================================================================

st.markdown("### ❓ Ask About the Loan")

question = st.text_input(
    "Example: Does this borrower have a gift letter? Are we compliant with FHA rules?",
    "What compliance issues exist for this loan application?"
)

# ============================================================================
# STEP 14: MAIN UI - RUN ASSESSMENT
# ============================================================================

if st.button("🔍 Run Assessment", type="primary"):
    
    if not assess_fannie_mae and not assess_fha:
        st.warning("Select at least one guideline!")
    else:
        st.markdown("---")
        
        # RUN FANNIE MAE ASSESSMENT
        if assess_fannie_mae:
            st.subheader("📊 Fannie Mae Assessment")
            
            state_fannie = {
                "question": question,
                "borrower_type": borrower_type,
                "gift": gift,
                "large_deposit": large_deposit,
                "case_id": case_id,
                "guideline": "Fannie Mae",
                "retrieved_docs": [],
                "citations": [],
                "compliance_flags": [],
                "answer": ""
            }
            
            result_fannie = underwriter_app.invoke(state_fannie)
            st.markdown(result_fannie["answer"])
        
        st.markdown("---")
        
        # RUN FHA ASSESSMENT
        if assess_fha:
            st.subheader("📊 FHA Assessment")
            
            state_fha = {
                "question": question,
                "borrower_type": borrower_type,
                "gift": gift,
                "large_deposit": large_deposit,
                "case_id": case_id,
                "guideline": "FHA",
                "retrieved_docs": [],
                "citations": [],
                "compliance_flags": [],
                "answer": ""
            }
            
            result_fha = underwriter_app.invoke(state_fha)
            st.markdown(result_fha["answer"])
