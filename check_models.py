import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fetch and print the list of active models for your account
models = client.models.list()
print("Available Groq Models:")
for m in models.data:
    print(f"- {m.id}")