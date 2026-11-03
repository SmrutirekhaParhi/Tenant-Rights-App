import time
import uuid
import re
import streamlit as st
from prefilter import classify_complaint
import rag
from extractor import extract_lease_entities, mask_pii_for_display
from agent import draft_legal_response

# Bulletproof bindings so ImportError can never happen
retrieve_relevant_rules = rag.retrieve_relevant_rules
process_and_store_pdf = rag.process_and_store_pdf

def delete_session_data(session_id="default"):
    if hasattr(rag, "delete_session_data"):
        return rag.delete_session_data(session_id)
    safe_name = f"lease_{re.sub(r'[^a-zA-Z0-9]', '', session_id)[:24]}"
    try:
        rag.chroma_client.delete_collection(name=safe_name)
    except Exception:
        pass
    return True

st.set_page_config(
    page_title="Tenant Rights Enforcer",
    page_icon="⚖️",
    layout="wide"
)
# Shrink metric value font size so long categories like "Maintenance & Habitability" never get cut off
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# 1. Assign a private per-user session ID so ChromaDB collections are never shared
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Header Section
st.title("⚖️ Global Tenant Rights & Lease Enforcer")
st.caption("Privacy-First Autonomous RAG Legal Advisor • Powered by Tesseract OCR, ChromaDB & Groq LPUs")
st.divider()

# Main Split-Screen Layout
col_input, col_output = st.columns([1.05, 1.35], gap="large")

with col_input:
    with st.container(border=True):
        st.subheader("📋 Case Intake & Lease Upload")
        
        st.markdown("**1. Property Jurisdiction**")
        c1, c2 = st.columns(2)
        with c1:
            country = st.selectbox(
                "Country",
                ["India", "United States", "United Kingdom", "Canada", "Australia"]
            )
        with c2:
            state = st.text_input(
                "State / Province / Region",
                value="Tamil Nadu",
                placeholder="e.g., Tamil Nadu, Karnataka"
            )

        st.markdown("**2. Lease or Legal Document (PDF)**")
        st.caption(
            "🔒 **Privacy Notice:** Your documents are deleted once they are processed. "

        )
        
        uploaded_file = st.file_uploader(
            "Upload your signed lease agreement (PDF)",
            type="pdf"
        )

        if uploaded_file is not None:
            if (
                "processed_file" not in st.session_state
                or st.session_state.processed_file != uploaded_file.name
            ):
                with st.spinner("Running OCR, indexing private ChromaDB collection & extracting parties..."):
                    num_chunks, first_last_text = process_and_store_pdf(
                        uploaded_file,
                        session_id=st.session_state.session_id
                    )
                    st.session_state.num_chunks = num_chunks
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.entities = extract_lease_entities(first_last_text)
                    
            st.success(
                f"✅ Indexed into {st.session_state.num_chunks} private segments. Raw PDF deleted from memory."
            )

            # Show only Landlord & Property Address verification (Tenant PII is never extracted)
            ent = st.session_state.get("entities", {})
            with st.expander("✅ Verify Extracted Landlord Details", expanded=True):
                st.caption("Tenant personal details are not extracted for privacy. Verify the Landlord details below:")
                v_landlord = st.text_input(
                    "Landlord Name",
                    value=ent.get("landlord_name", "Mr. Krishna Ranganathan")
                )
                v_address = st.text_input(
                    "Flat / Property Address",
                    value=ent.get("property_address", "Flat J-402, Abode Valley, Potheri, Chennai")
                )
                
                verified_details = {
                    "landlord_name": v_landlord,
                    "property_address": v_address
                }
                
                if st.button("🗑️ Purge Session & Delete Document Data", use_container_width=True):
                    delete_session_data(st.session_state.session_id)
                    for key in ["processed_file", "num_chunks", "entities", "last_result"]:
                        st.session_state.pop(key, None)
                    st.toast("All vector embeddings and extracted details wiped from memory.", icon="🗑️")
                    st.rerun()
        else:
            verified_details = {}

        st.markdown("**3. Describe the Housing Issue**")
        user_complaint = st.text_area(
            "Tenant Complaint:",
            height=100,
            placeholder="e.g., Water pipe is broken or Chimney is not working and landlord is ignoring calls."
        )

        generate_btn = st.button(
            "⚖️ Analyze Lease & Draft Legal Notice",
            type="primary",
            use_container_width=True
        )

with col_output:
    st.subheader("📑 Legal Intelligence & Demand Notice")
    
    if generate_btn:
        if not user_complaint or not state:
            st.warning("⚠️ Please fill in both the State/Region and the complaint description.")
        elif uploaded_file is None:
            st.warning("⚠️ Please upload a Lease PDF first so the AI can verify your clauses.")
        else:
            with st.spinner("Classifying dispute, retrieving clauses, and drafting notice..."):
                classification = classify_complaint(user_complaint)
                retrieved_clauses = retrieve_relevant_rules(
                    user_complaint,
                    classification=classification,
                    session_id=st.session_state.session_id
                )
                
                # Measure pure Groq LLM generation latency
                groq_start = time.time()
                draft = draft_legal_response(
                    user_complaint,
                    classification,
                    retrieved_clauses,
                    country,
                    state,
                    verified_details
                )
                groq_elapsed = time.time() - groq_start
                
                st.session_state.last_result = {
                    "classification": classification,
                    "clauses": retrieved_clauses,
                    "draft": draft,
                    "elapsed": groq_elapsed
                }

    if "last_result" in st.session_state:
        res = st.session_state.last_result
        
        m1, m2, m3 = st.columns([1.6, 1, 1])
        with m1:
            st.metric("Dispute Category", res["classification"])
        with m2:
            st.metric("Lease Clauses Matched", f"{len(res['clauses'])} Segments")
        with m3:
            st.metric("Groq LPU Latency", f"{res['elapsed']:.2f} sec")

        with st.expander("🔍 Inspect Retrieved Lease Clauses (PII Masked for Privacy)", expanded=False):
            for idx, clause in enumerate(res["clauses"], 1):
                st.markdown(f"**Segment {idx}:** {mask_pii_for_display(clause)}")
                if idx < len(res["clauses"]):
                    st.divider()

        with st.container(border=True):
            st.markdown(res["draft"])
            st.divider()
            st.download_button(
                label="📥 Download Formal Demand Notice (.txt)",
                data=res["draft"],
                file_name="Legal_Demand_Notice.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.caption("⚠️ *Disclaimer: This tool provides automated legal information grounded in your lease and statutory text, not formal legal representation.*")
    else:
        with st.container(border=True):
            st.info(
                "👈 Upload your lease PDF, confirm the auto-extracted party details in the verification box, "
                "and describe your housing issue to generate a statute-grounded legal demand notice."
            )
            
if __name__ == "__main__":
    import os
    import sys
    from streamlit.web import cli as stcli

    # Only launch the server if Streamlit is not already running
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())