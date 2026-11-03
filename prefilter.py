import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Expanded synthetic training dataset to teach the Bouncer
# Added specific keywords to capture appliance, ventilation, and communication issues
data = {
    'text': [
        # Maintenance & Habitability
        "Chimney issue and landlord is not picking up the call.",
        "The kitchen exhaust fan and chimney are broken and owner is ignoring my texts.",
        "The geyser in the bathroom is not heating water and needs repair.",
        "Severe water seepage and dampness on the bedroom walls during monsoon.",
        "The AC is leaking water and cooling has stopped completely.",
        "Bathroom plumbing is blocked and dirty water is overflowing.",
        "The electrical wiring is sparking and the main switchboard is damaged.",
        "Water motor is broken and there is no water supply in the flat.",
        
        # Financial Dispute
        "My landlord won't return my security deposit after vacating the flat.",
        "Owner is deducting two months of advance for repainting and deep cleaning.",
        "They are charging me a random convenience fee and extra society maintenance.",
        "Landlord is demanding a 20 percent rent hike in the middle of the lease.",
        "Owner is overcharging for electricity per unit on the sub-meter without showing the bill.",
        "Refusing to refund the token advance even though the agreement was not signed.",
        "Deducting money from my security deposit for old wear and tear.",
        "Landlord is asking me to pay for major structural plumbing repairs.",
        
        # Illegal Eviction & Lockout
        "I got a sudden notice to vacate the apartment in 3 days without reason.",
        "The landlord changed the flat locks without telling me while I was at college.",
        "Owner disconnected the electricity and water supply to force us to vacate.",
        "Landlord is threatening to throw our luggage out if we do not leave by tomorrow.",
        "Asking us to vacate immediately without serving the one month notice period.",
        "The owner locked the main gate and is not letting tenants enter the building.",
        "Terminating the rental agreement early without any lease violation on our part.",
        "Forcing us to leave the flat overnight because his relatives are moving in.",
        
        # Harassment & Privacy
        "The landlord keeps entering my flat using a spare key without prior permission.",
        "Owner is harassing us about friends and guests visiting during the daytime.",
        "The society watchman and landlord are imposing unfair curfew timings on tenants.",
        "Landlord is verbally abusing us on the phone and making threatening visits.",
        "Owner refuses to allow my parents to stay overnight in our rented apartment.",
        "Constant unannounced inspections by the landlord every weekend.",
        "Harassing female tenants with moral policing and unnecessary CCTV surveillance.",
        "Blocking our car and bike parking slots despite it being included in the lease."
    ],
    'category': [
        "Maintenance & Habitability", "Maintenance & Habitability", "Maintenance & Habitability", "Maintenance & Habitability",
        "Maintenance & Habitability", "Maintenance & Habitability", "Maintenance & Habitability", "Maintenance & Habitability",
        "Financial Dispute", "Financial Dispute", "Financial Dispute", "Financial Dispute",
        "Financial Dispute", "Financial Dispute", "Financial Dispute", "Financial Dispute",
        "Illegal Eviction & Lockout", "Illegal Eviction & Lockout", "Illegal Eviction & Lockout", "Illegal Eviction & Lockout",
        "Illegal Eviction & Lockout", "Illegal Eviction & Lockout", "Illegal Eviction & Lockout", "Illegal Eviction & Lockout",
        "Harassment & Privacy", "Harassment & Privacy", "Harassment & Privacy", "Harassment & Privacy",
        "Harassment & Privacy", "Harassment & Privacy", "Harassment & Privacy", "Harassment & Privacy"
    ]
}
df = pd.DataFrame(data)

# 2. Build the Logarithmic Pre-Filter Pipeline
bouncer_model = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression()
)

# 3. Train the model on the expanded data
bouncer_model.fit(df['text'], df['category'])

# 4. Define the filtering function
def classify_complaint(user_input):
    prediction = bouncer_model.predict([user_input])[0]
    return prediction

# Run a quick test if this file is executed directly
if __name__ == "__main__":
    test_complaint = "Chimney issue and landlord is not picking up the call."
    print(f"User Complaint: '{test_complaint}'")
    print(f"Bouncer Classification: {classify_complaint(test_complaint)}")