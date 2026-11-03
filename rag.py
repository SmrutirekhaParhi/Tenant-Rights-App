import fitz  # PyMuPDF
import chromadb
import pytesseract
from PIL import Image
import io
import re

# Point Python to the local OCR engine executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# In-memory ChromaDB client (never writes user PDFs to disk)
chroma_client = chromadb.Client()

def _get_session_collection(session_id):
    """Creates an isolated, private collection per user session to prevent data leaks."""
    safe_name = f"lease_{re.sub(r'[^a-zA-Z0-9]', '', session_id)[:24]}"
    return chroma_client.get_or_create_collection(name=safe_name)

def clean_ocr_text(text):
    """Removes OCR noise, stray pipes, and excessive dots/dashes from scanned leases."""
    text = text.replace('\n', ' ')
    text = re.sub(r'[|—_~=`]+', ' ', text)
    text = re.sub(r'[\.\-]{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_and_store_pdf(uploaded_file, session_id="default"):
    """
    Processes the PDF in memory, stores chunks in a private session collection,
    and returns (num_chunks, first_and_last_page_text) for one-time entity extraction.
    """
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    page_texts = []
    for page in doc:
        text = page.get_text().strip()
        if len(text) < 50:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img)
        page_texts.append(clean_ocr_text(text))
        
    doc.close()
    del pdf_bytes  # Explicitly release raw PDF bytes after OCR

    full_text = " ".join(page_texts)
    
    # Combine ONLY first and last page for privacy-preserving party extraction
    if len(page_texts) > 1:
        first_last_text = page_texts[0] + " \n " + page_texts[-1]
    elif len(page_texts) == 1:
        first_last_text = page_texts[0]
    else:
        first_last_text = ""

    chunk_size = 1000
    overlap = 200
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(full_text), step):
        chunks.append(full_text[i:i + chunk_size])

    collection = _get_session_collection(session_id)
    try:
        existing = collection.get()
        if existing and existing['ids']:
            collection.delete(ids=existing['ids'])
    except Exception:
        pass

    valid_chunks = 0
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:
            collection.add(
                documents=[chunk],
                metadatas=[{"source": "upload", "chunk_index": i}],
                ids=[f"chunk_{i}"]
            )
            valid_chunks += 1

    return valid_chunks, first_last_text

def retrieve_relevant_rules(complaint, classification="", session_id="default", n_results=3):
    """Expands the query with legal keywords and searches the user's private collection."""
    collection = _get_session_collection(session_id)
    if collection.count() == 0:
        return ["No document uploaded yet. Please upload a Lease or Law PDF."]

    # Query expansion so short complaints ("Water pipe is broken") match formal lease terms
    expanded_query = f"{complaint} {classification} repair maintenance fixtures landlord tenant obligation clause"
    
    results = collection.query(
        query_texts=[expanded_query],
        n_results=min(n_results, collection.count())
    )

    if not results['documents'] or not results['documents'][0]:
        return ["No relevant rules found in the document."]

    return results['documents'][0]

def delete_session_data(session_id="default"):
    """Permanently deletes the user's vector collection from memory."""
    safe_name = f"lease_{re.sub(r'[^a-zA-Z0-9]', '', session_id)[:24]}"
    try:
        chroma_client.delete_collection(name=safe_name)
        return True
    except Exception:
        return False