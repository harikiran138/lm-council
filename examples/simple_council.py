import asyncio
from lm_council import LanguageModelCouncil
from dotenv import load_dotenv
import os

async def main():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY not found in environment variables.")

    print("Initializing Council...")
    lmc = LanguageModelCouncil(
        models=[
            "openai/gpt-3.5-turbo",
        ],
    )

    print("Executing Council...")
    try:
        completion, judgment = await lmc.execute("Say hello.")
        print("Execution successful!")
        print("Completion:", completion)
        print("Judgment:", judgment)
    except Exception as e:
        print(f"Execution failed (expected if no API key): {e}")

if __name__ == "__main__":
    asyncio.run(main())
