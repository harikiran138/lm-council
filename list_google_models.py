import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/models"

response = requests.get(
    base_url,
    params={"key": api_key}
)

if response.status_code == 200:
    models = response.json().get("models", [])
    print("Available Models:")
    for m in models:
        print(f"- {m['name']}")
        if "supportedGenerationMethods" in m:
            print(f"  Methods: {m['supportedGenerationMethods']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
