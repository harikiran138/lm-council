import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from lm_council import LanguageModelCouncil

# Load environment variables
load_dotenv()

st.set_page_config(page_title="LM Council", layout="wide")

st.title("Language Model Council")

# Sidebar Configuration
st.sidebar.header("Configuration")

api_key = os.getenv("OPENROUTER_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not api_key and not gemini_api_key:
    st.sidebar.error("No API keys found in environment variables.")
    st.stop()

if api_key:
    st.sidebar.success("OpenRouter API Key found.")
if gemini_api_key:
    st.sidebar.success("Gemini API Key found.")

# Model Selection
all_models = []

if gemini_api_key:
    all_models.extend([
        "gemini-2.0-flash-exp",
    ])

if api_key:
    all_models.extend([
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3-70b-instruct",
    ])

# Default selection
default_selection = []
if gemini_api_key:
    default_selection.append("gemini-2.0-flash-exp")
if api_key:
    default_selection.append("openai/gpt-3.5-turbo")

selected_models = st.sidebar.multiselect(
    "Select Models for Council",
    options=all_models,
    default=default_selection
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# Main Area
user_prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Run Council"):
    if not user_prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Council is deliberating..."):
            try:
                # Initialize Council
                lmc = LanguageModelCouncil(
                    models=selected_models,
                    api_key=api_key,
                    google_api_key=gemini_api_key
                )
                
                # Run execution
                # We need to run the async method in a sync Streamlit app
                async def run_council():
                    return await lmc.execute(user_prompt)

                completions_df, judgments_df = asyncio.run(run_council())
                
                st.success("Execution Complete!")
                
                st.subheader("Completions")
                st.dataframe(completions_df)
                
                st.subheader("Judgments")
                st.dataframe(judgments_df)
                
                # Display individual responses nicely
                st.subheader("Detailed Responses")
                for index, row in completions_df.iterrows():
                    with st.expander(f"Response from {row['model']}"):
                        st.markdown(row['completion_text'])
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
