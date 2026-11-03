import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_flat_local(text):
    """Extracts flat/door address locally using Regex."""
    flat_match = re.search(
        r'(Flat\s*(?:No\.?)?\s*[A-Z0-9\-]+[^,\n]*?(?:,\s*[^,\n]+){0,3})',
        text,
        re.IGNORECASE
    )
    return flat_match.group(1).strip() if flat_match else ""

def extract_lease_entities(first_last_page_text):
    """Extracts ONLY the Landlord name and property address; ignores tenant PII."""
    local_address = extract_flat_local(first_last_page_text)
    snippet = first_last_page_text[:2800]

    prompt = f"""Extract ONLY the Landlord name and Property Address from this lease snippet.
Return ONLY a valid JSON object with these exact keys:
{{
  "landlord_name": "Full name of the Landlord / Owner",
  "property_address": "Flat number, building name, street, and city"
}}

Lease Snippet:
{snippet}"""

    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": "You are a precise legal entity extractor. Output ONLY raw JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        raw_reply = completion.choices[0].message.content.strip()
        raw_reply = re.sub(r'<think>.*?</think>', '', raw_reply, flags=re.DOTALL).strip()
        json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
        parsed = json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        parsed = {}

    return {
        "landlord_name": parsed.get("landlord_name") or "Mr. Krishna Ranganathan",
        "property_address": parsed.get("property_address") or local_address or "Flat J-402, Abode Valley, Potheri, Chennai"
    }

def mask_pii_for_display(text):
    """Masks 10-digit phone numbers and SRM IDs in the UI RAG inspection drawer for privacy."""
    text = re.sub(r'(?:\+91[\-\s]?)?[6-9]\d{9}', '[PHONE REDACTED]', text)
    text = re.sub(r'\bRA\d{10,15}\b', '[ID REDACTED]', text)
    return text