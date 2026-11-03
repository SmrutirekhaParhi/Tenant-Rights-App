import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

STATUTE_BANK = {
    "tamil nadu": (
        "Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017 "
        "(TNRRRLT Act, 2017). Under Section 15, the landlord is statutorily responsible for structural "
        "repairs, main plumbing, external electrical wiring, and restoring essential fixtures provided "
        "in the lease. If the landlord fails to repair after notice, the tenant may approach the Rent Authority."
    ),
    "india_default": (
        "Model Tenancy Act, 2021 and Section 108(f) of the Transfer of Property Act, 1882, which obligate "
        "the lessor to maintain essential services and structural habitability."
    )
}

def enforce_output_guardrails(text, verified_details):
    """
    Guarantees active 2017 Act citation, exact landlord name, and forces the
    exact 3-line placeholder signature at the bottom of the letter.
    """
    # 1. Strip any <think>...</think> reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Hard-replace the repealed 1960 Tamil Nadu Act if the model hallucinates it
    text = re.sub(
        r'Tamil Nadu Buildings \(Lease and Rent Control\) Act,?\s*1960',
        'Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017 (Section 15)',
        text,
        flags=re.IGNORECASE
    )

    # 3. Replace Landlord/Address placeholders if any appear
    landlord_name = verified_details.get("landlord_name", "Landlord")
    address = verified_details.get("property_address", "")
    text = re.sub(r'\[Landlord Name\]|\[Landlord\'?s? Name\]', landlord_name, text, flags=re.IGNORECASE)
    text = re.sub(r'\[Property Address\]|\[Flat Number\]', address, text, flags=re.IGNORECASE)

    # 4. Hard-force the exact 3-line placeholder block at the end of the letter
    # Double trailing spaces force Markdown to render each placeholder on its own line
    exact_signoff = "Regards,  \n[name]  \n[contact no.-]  \n[email-]"
    
    if re.search(r'(?:Regards|Sincerely|Yours truly|Best regards),?.*$', text, flags=re.IGNORECASE | re.DOTALL):
        text = re.sub(
            r'(?:Regards|Sincerely|Yours truly|Best regards),?.*$',
            exact_signoff,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    else:
        text = text.rstrip() + "\n\n" + exact_signoff

    return text

def draft_legal_response(user_complaint, classification, retrieved_clauses, country, state, verified_details):
    context_text = "\n---\n".join(retrieved_clauses)
    statute_context = STATUTE_BANK["tamil nadu"] if "tamil" in state.lower() else STATUTE_BANK["india_default"]

    landlord = verified_details.get("landlord_name", "Landlord")
    address = verified_details.get("property_address", "")

    system_prompt = f"""You are an elite Tenant Rights Legal Advisor operating in {state}, {country}.
Follow these strict rules:
1. ACTIVE STATUTE ONLY: You MUST cite: {statute_context}. NEVER cite the repealed 1960 Act.
2. LANDLORD & PRIVACY PLACEHOLDERS: Address the letter to '{landlord}' regarding property '{address}'. Sign the bottom of the letter using ONLY:
Regards,
[name]
[contact no.-]
[email-]
3. SPECIFIC CLAUSE GROUNDING: Quote or reference specific items found in the Retrieved Lease Clauses (e.g., Clause 5, Modeler Kitchen, Water Heater, Chimney, Exhaust, or Air Conditioner).
4. SAFE REMEDIES: Never advise unilateral rent withholding without prior statutory notice."""

    user_prompt = f"""
Confirmed Lease Details:
- Landlord: {landlord}
- Property Address: {address}

Tenant's Complaint ({classification}): "{user_complaint}"

Retrieved Lease Clauses:
{context_text}

Format your response in clean Markdown with EXACTLY these two headings:
### Legal Analysis
(3-4 sentences citing the exact 2017 Act section, referencing the specific clause/fixture from the retrieved lease segments, and stating the lawful remedy.)

### Suggested Demand Letter
(A formal demand email addressed to Dear {landlord}, referencing {address}, giving a 48-hour repair deadline, and signed at the bottom with:
Regards,
[name]
[contact no.-]
[email-])
"""

    completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
    )

    raw_output = completion.choices[0].message.content
    return enforce_output_guardrails(raw_output, verified_details)