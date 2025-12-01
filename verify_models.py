import asyncio
import os
from dotenv import load_dotenv
from lm_council import LanguageModelCouncil

load_dotenv()

google_models = [
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-2.0-flash-exp",
]

openrouter_models = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "google/gemini-pro",
    "google/gemini-flash-1.5",
]

async def test_model(lmc, model):
    print(f"Testing {model}...", end=" ", flush=True)
    try:
        # We use a very simple prompt and try to get a text completion directly
        # to avoid the overhead of the full council execution if possible, 
        # but execute is safer to test the full path.
        # Let's use get_text_completion directly to be faster and isolate the model check.
        await lmc.get_text_completion("Hi", model)
        print("✅ OK")
        return True, "OK"
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    lmc = LanguageModelCouncil(
        models=[], # We will pass model explicitly
        api_key=api_key,
        google_api_key=gemini_api_key
    )

    working_models = []
    
    with open("results.txt", "w") as f:
        f.write("--- Testing Google Models ---\n")
        print("--- Testing Google Models ---")
        for model in google_models:
            success, msg = await test_model(lmc, model)
            f.write(f"{model}: {msg}\n")
            if success:
                working_models.append(model)

        f.write("\n--- Testing OpenRouter Models ---\n")
        print("\n--- Testing OpenRouter Models ---")
        for model in openrouter_models:
            success, msg = await test_model(lmc, model)
            f.write(f"{model}: {msg}\n")
            if success:
                working_models.append(model)

        f.write("\n\n=== Working Models List ===\n")
        f.write(str(working_models))
    
    print("\n\n=== Working Models List ===")
    print(working_models)

if __name__ == "__main__":
    asyncio.run(main())
