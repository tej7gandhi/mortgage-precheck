"""
MORTGAGE UNDERWRITING APP
Assess loans against Fannie Mae & FHA guidelines simultaneously
"""

import streamlit as st
import os
import io
from typing import TypedDict, List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json
from groq import Groq

# ============================================================================
# STEP 1: CONFIGURATION - API KEYS & DATABASE CONNECTION
# ============================================================================

st.set_page_config(page_title="Mortgage Underwriter", page_icon="🏦", layout="wide")
st.title("🏦 Dual Guideline Mortgage Underwriter")
st.caption("Fannie Mae + FHA Assessment")

# Read default API keys from environment (or leave blank for user input)
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "")
DEFAULT_QDRANT_KEY = os.getenv("QDRANT_KEY", "")
DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# ============================================================================
# STEP 2: SIDEBAR - USER INPUTS FOR LOAN SCENARIO
# ============================================================================

st.sidebar.subheader("🔑 Groq API Key (for auto-fill)")
groq_key_input = st.sidebar.text_input(
    "Groq API Key",
    value=DEFAULT_GROQ_KEY,
    type="password",
    key="groq_key_input"
)
GROQ_KEY = groq_key_input if groq_key_input else DEFAULT_GROQ_KEY

st.sidebar.subheader("📥 Auto-Fill from Loan Application")
st.sidebar.caption("Upload the application and Groq will read the fields below for you.")

application_file = st.sidebar.file_uploader(
    "Loan application PDF",
    type=["pdf"],
    key="application_uploader"
)

if application_file and st.sidebar.button("✨ Extract & Auto-Fill", key="extract_button"):
    if not GROQ_KEY:
        st.sidebar.error("❌ Enter a Groq API key above first.")
    else:
        with st.sidebar.status("Reading application and extracting fields..."):
            try:
                reader = PdfReader(io.BytesIO(application_file.getvalue()))
                application_text = "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                ).strip()

                if not application_text:
                    st.sidebar.warning(
                        "No extractable text found. (Scanned/image-only PDFs "
                        "need OCR first — this reader only reads text layers.)"
                    )
                else:
                    groq_client = Groq(api_key=GROQ_KEY)

                    extraction_prompt = f"""You are extracting structured fields from a mortgage loan application.
Read the application text below and return ONLY a JSON object with exactly these keys:

- "borrower_type": one of "Salaried", "Self-Employed", "Retired" (best match based on stated employment/income type)
- "gift": "Yes" or "No" (whether any portion of the down payment comes from a family gift)
- "large_deposit": "Yes" or "No" (whether bank statements show a large deposit exceeding 50% of monthly income)
- "case_id": the loan number / case ID / application number as a string (or "" if none found)

If a field cannot be determined from the text, make your best reasonable guess for borrower_type/gift/large_deposit rather than leaving it blank, and use "" only for case_id if truly absent.

APPLICATION TEXT:
{application_text[:12000]}

Return ONLY the JSON object, no other text."""

                    completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": extraction_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0,
                    )

                    extracted = json.loads(completion.choices[0].message.content)

                    # Pre-fill the manual fields below by setting their widget state,
                    # then rerun so the widgets pick up these values.
                    if extracted.get("borrower_type") in ["Salaried", "Self-Employed", "Retired"]:
                        st.session_state["borrower_type_input"] = extracted["borrower_type"]
                    if extracted.get("gift") in ["Yes", "No"]:
                        st.session_state["gift_input"] = extracted["gift"]
                    if extracted.get("large_deposit") in ["Yes", "No"]:
                        st.session_state["large_deposit_input"] = extracted["large_deposit"]
                    if extracted.get("case_id"):
                        st.session_state["case_id_input"] = extracted["case_id"]

                    st.sidebar.success("✅ Fields extracted — updating form below.")
                    st.rerun()

            except json.JSONDecodeError:
                st.sidebar.error("❌ Groq returned a response that wasn't valid JSON. Try again.")
            except Exception as e:
                st.sidebar.error(f"❌ Extraction failed: {e}")

st.sidebar.divider()
st.sidebar.header("📋 Loan Details")
st.sidebar.caption("Auto-filled fields above can still be reviewed or corrected here.")

# Borrower info
borrower_type = st.sidebar.selectbox(
    "Borrower Type",
    ["Salaried", "Self-Employed", "Retired"],
    key="borrower_type_input"
)

# Down payment info
gift = st.sidebar.radio(
    "Family Gift for Down Payment?",
    ["No", "Yes"],
    key="gift_input"
)

# Deposit info
large_deposit = st.sidebar.radio(
    "Large Deposits (>50% monthly income)?",
    ["No", "Yes"],
    key="large_deposit_input"
)

# Case identifier
case_id = st.sidebar.text_input(
    "Case ID / Loan Number",
    "loan-12345",
    key="case_id_input"
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
    
    # Scope the search to only documents tagged with the current guideline,
    # so Fannie Mae and FHA queries can't return each other's chunks.
    guideline_filter = None
    if state.get('guideline'):
        guideline_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.guideline",
                    match=MatchValue(value=state['guideline'])
                )
            ]
        )

    # Search both knowledge bases (curated_db is searched unfiltered, since
    # it's a shared cross-guideline knowledge base rather than raw guideline text)
    mortgage_results = mortgage_db.similarity_search_with_score(
        search_query, k=3, filter=guideline_filter
    )
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

st.markdown("### 📄 Add Documents to Knowledge Base")
st.caption("Upload a guideline PDF once, and it becomes searchable for every future assessment.")

col1, col2 = st.columns([1, 2])
with col1:
    upload_guideline = st.selectbox(
        "This document is:",
        ["Fannie Mae", "FHA"],
        key="upload_guideline"
    )
with col2:
    citation_label = st.text_input(
        "Citation label (optional)",
        placeholder="e.g. Fannie Mae Selling Guide B1-1-01",
        key="citation_label"
    )

uploaded_files = st.file_uploader(
    "Upload guideline PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key="guideline_uploader"
)

if uploaded_files:
    st.write(f"Ready to add **{len(uploaded_files)}** file(s) as **{upload_guideline}** guidance:")
    for f in uploaded_files:
        st.write(f"  • {f.name}")

    if st.button("➕ Add to Knowledge Base", type="primary"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_chunks = []
        failed_files = []

        with st.spinner("Reading, chunking, and embedding documents... this may take a minute."):
            for uploaded_file in uploaded_files:
                try:
                    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    full_citation = citation_label if citation_label else f"{upload_guideline} Guideline Document"

                    for page_num, page in enumerate(reader.pages, start=1):
                        page_text = page.extract_text() or ""
                        if not page_text.strip():
                            continue  # skip blank/scanned pages with no extractable text

                        page_doc = Document(
                            page_content=page_text,
                            metadata={
                                "guideline": upload_guideline,   # used by the retrieval filter
                                "citation": full_citation,
                                "page": page_num,
                                "source": uploaded_file.name,
                                "approved_by": "streamlit-upload",
                            }
                        )
                        all_chunks.extend(splitter.split_documents([page_doc]))

                except Exception as e:
                    failed_files.append((uploaded_file.name, str(e)))

            if all_chunks:
                try:
                    mortgage_db.add_documents(all_chunks)
                    st.success(
                        f"✅ Added {len(all_chunks)} chunks from {len(uploaded_files) - len(failed_files)} "
                        f"file(s) to the **{upload_guideline}** knowledge base."
                    )
                except Exception as e:
                    st.error(f"❌ Failed to upload chunks to Qdrant: {e}")
            else:
                st.warning("No extractable text was found in the uploaded file(s). "
                            "(Scanned/image-only PDFs need OCR first — this uploader only reads text layers.)")

            for fname, err in failed_files:
                st.error(f"❌ Failed to process {fname}: {err}")

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
