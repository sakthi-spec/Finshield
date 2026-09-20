import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()

# Read the API key without exposing it in the code
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing.")

# Create the Gemini client
client = genai.Client(api_key=api_key)

# Send a simple request to the LLM
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain what a bank statement is in one simple sentence."
)

print("----- GEMINI RESPONSE -----")
print(interaction.output_text)