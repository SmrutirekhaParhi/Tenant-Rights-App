# 🏛️ Tenant Rights & Lease Enforcer AI Assistant

A privacy-first, statute-grounded LegalTech RAG application designed to analyze residential lease agreements, classify tenant disputes, and generate legally compliant demand notices enforced by regional statutory frameworks.

---

## 🚀 Key Features

* **Privacy-First Hybrid PII Extraction:** Combines local Regex pattern matching for property addresses and targeted LLM snippet passes for landlord names, ensuring sensitive user details never leak.
* **Deterministic Statute Grounding:** Overcomes standard LLM hallucinations by using a Python post-processing guardrail (`enforce_output_guardrails`) that blocks obsolete laws and strictly enforces active regional legislation (e.g., *Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017*).
* **Supervised Intent Routing:** Utilizes a classical machine learning classifier (`TfidfVectorizer` + `LogisticRegression`) in `prefilter.py` to instantly categorize natural-language tenant complaints into structured legal taxonomy buckets.
* **Semantic Vector Retrieval (RAG):** Ingests digital PDFs via PyMuPDF or scanned stamp papers via Tesseract OCR, storing dense vector embeddings locally in volatile, session-isolated ChromaDB collections (`lease_{session_id}`).
* **High-Speed LPU Inference:** Powered by Groq LPUs hosting open-weights Qwen 27B for sub-2-second response generation.
* **Zero-Persistence Data Lifecycle:** Includes volatile session memory isolation and a one-click data purge mechanism to align with modern privacy standards.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.10+
* **Frontend Dashboard:** Streamlit
* **Vector Database & Embeddings:** ChromaDB (local MiniLM-class transformer embeddings)
* **OCR & Document Processing:** PyMuPDF (`fitz`), Tesseract OCR (300 DPI)
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `TfidfVectorizer`)
* **LLM & Inference:** Groq API (Qwen 27B model)

---

## 📊 System Architecture Pipeline

```text
[ User PDF Lease + Natural Language Complaint ]
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 1. Document Ingestion & OCR (rag.py)         │
│  • PyMuPDF & Tesseract OCR (300 DPI)         │
│  • Sliding-Window Chunking (1000/200 overlap)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. Local PII Extraction & Verification       │
│  • Local Regex Address Extraction            │
│  • Targeted Landlord Name Snippet Pass       │
│  • Human-in-the-Loop UI Verification Box     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 3. Private Vector Indexing (ChromaDB)        │
│  • Session-Isolated Collections              │
│  • Cosine Similarity Semantic Search         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. Intent Routing & RAG Retrieval            │
│  • Scikit-Learn Classifier (prefilter.py)    │
│  • Contextual Clause Extraction              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 5. Groq LPU Inference & Guardrails           │
│  • Qwen 27B Legal Analysis                   │
│  • Python Post-Processing Statute Guardrails │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 6. Interactive Dashboard (Streamlit)         │
│  • Telemetry Metrics & RAG Inspection Drawer │
│  • Plain-Text Download & Session Wipe        │
└──────────────────────────────────────────────┘

##Project Structure

├── app.py              # Streamlit front-end interface and dashboard
├── agent.py            # Groq LPU integration and output guardrails
├── rag.py              # Document ingestion, OCR, and ChromaDB vector store
├── extractor.py        # Local Regex and snippet-based PII extraction
├── prefilter.py        # Scikit-learn intent classification model
├── run_app.bat         # Windows quick-launch script
├── requirements.txt    # Project Python dependencies
└── .gitignore          # Excludes venv, .env, and local vector DBs
