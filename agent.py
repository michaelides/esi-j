import os
import random
import streamlit as st # Import streamlit for caching
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from tools import get_all_tools
from dotenv import load_dotenv

load_dotenv()


# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# --- Constants ---
SUGGESTED_PROMPT_COUNT = 4

# --- Global Settings ---
# @st.cache_resource # Cache the LLM and embedding model initialization - TEMPORARILY REMOVED FOR DIAGNOSIS
def initialize_settings():
    """Initializes LlamaIndex settings with Gemini LLM and Embedding model."""
    print("Attempting to initialize settings...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    print(f"  GOOGLE_API_KEY value: {google_api_key}")
    if not google_api_key:
        # raise ValueError("GOOGLE_API_KEY not found in environment variables.") # Keep running for now
        print("  WARNING: GOOGLE_API_KEY not found in environment variables!")
        # Attempt to set a placeholder or handle gracefully if possible,
        # For now, let it proceed to see if Gemini init handles it with its own error.
    
    # Use Google Generative AI Embeddings
    try:
        Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=google_api_key)
        print(f"  Settings.embed_model initialized: {type(Settings.embed_model)}")
    except Exception as e:
        print(f"  Error initializing Settings.embed_model: {e}")

    # Use a potentially more stable model name and set a default temperature
    # The temperature can be overridden later based on the slider
    try:
        Settings.llm = Gemini(model_name="models/gemini-2.5-flash-preview-05-20", # Ensuring this is the problem model
                              api_key=google_api_key,
                              temperature=0.7) # Default temperature
        print(f"  Settings.llm initialized: {type(Settings.llm)}")
        if hasattr(Settings.llm, 'model_name'):
            print(f"  Settings.llm.model_name: {Settings.llm.model_name}")
        elif hasattr(Settings.llm, 'model'): # Some LLM client libs use 'model'
             print(f"  Settings.llm.model: {Settings.llm.model}")
        else:
            print("  Settings.llm does not have a 'model_name' or 'model' attribute to display.")

    except Exception as e:
        print(f"  Error initializing Settings.llm (Gemini): {e}")
        # Fallback to a default LLM or raise error if critical
        # For diagnosis, we'll print the error and let it potentially fail downstream
        # if Settings.llm is not correctly set.
        Settings.llm = None # Explicitly set to None if init fails

    print("LLM settings initialization attempt complete.")

# --- Greeting Generation ---
def generate_llm_greeting() -> str:
    """Generates a dynamic greeting message using the configured LLM."""
    static_fallback = "Hello! I'm ESI, your AI assistant for dissertation support. How can I help you today?"
    try:
        # Settings.llm is guaranteed to be initialized by app.py before this function is called
        llm = Settings.llm
        if not isinstance(llm, LLM): # Basic check
             print("Warning: LLM not configured correctly for greeting generation.")
             return static_fallback

        # Use a simple prompt for a greeting
        # Using 'complete' for a single, non-chat generation
        prompt = """Generate a single, short, friendly, and welcoming greeting message (1-2 sentences) 
        for a user interacting with an AI assistant named ESI designed to help with university dissertations. 
        Mention ESI by name. Provide only the greeting itself, and offer help to the user.
        """
        response = llm.complete(prompt)
        greeting = response.text.strip()

        # Basic validation
        if not greeting or len(greeting) < 10:
            print("Warning: LLM generated an empty or too short greeting. Falling back.")
            return static_fallback

        print(f"Generated LLM Greeting: {greeting}")
        return greeting

    except Exception as e:
        print(f"Error generating LLM greeting: {e}. Falling back to static message.")
        return static_fallback


# --- Unified Agent Definition ---
@st.cache_resource # Cache the agent creation
def create_unified_agent():
    """
    Creates a unified agent with all available tools and a comprehensive system prompt.
    """
    print("Initializing unified agent for new session (LLM settings should already be done)...")
    all_tools = get_all_tools() # Get all tools from tools.py

    if not all_tools:
        print("CRITICAL: No tools were loaded by get_all_tools(). Unified agent may be non-functional.")

    try:
        with open("esi_agent_instruction.md", "r") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        print("CRITICAL: esi_agent_instruction.md not found. Unified agent will use a fallback prompt.")
        system_prompt = "You are a helpful AI assistant. Use the provided tools to answer the user's questions."
    except Exception as e:
        print(f"CRITICAL: Error loading esi_agent_instruction.md: {e}. Unified agent will use a fallback prompt.")
        system_prompt = "You are a helpful AI assistant. Use the provided tools to answer the user's questions."

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=all_tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True  # Set to False for production
    )
    unified_agent_runner = AgentRunner(agent_worker)
    print(f"Unified agent created with {len(all_tools)} tools.")
    return unified_agent_runner

# --- Suggested Prompts ---
DEFAULT_PROMPTS = [
    "Find recent papers on qualitative data analysis methods.",
    "Explain the structure of a typical literature review.",
    "What are common challenges students face in writing?",
    "Search for university policies on dissertation submission deadlines (uses RAG).",
]

def generate_suggested_prompts(chat_history):
    """
    Generates concise suggested prompts.
     """
     # --- LLM-based Generation ---
    try:
        # Settings.llm is guaranteed to be initialized by app.py before this function is called
        llm = Settings.llm

        # Create context from the last few messages (e.g., last 4)
        context_messages = chat_history[-4:] # Get last 4 messages
        context_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in context_messages])

        # Construct the prompt for the LLM
        prompt = f"""Given the following recent conversation context between a User and an AI Assistant (ESI) helping with university dissertations:
--- CONTEXT START ---
{context_str}
--- CONTEXT END ---

Generate exactly {SUGGESTED_PROMPT_COUNT} concise and relevant follow-up responses or prompts (each under 15 words) that the User might respond with next. 
The prompts should be directly related to the conversation topics or typical dissertation support tasks. 

Output ONLY the prompts, each on a new line, without any numbering, bullet points, or introductory text.
Example:
I am interested in topic K.
I like theory L.
Find papers on topic X.
Explain concept Y.
How do I structure section Z?
Summarize the key points about W.
"""

        print("Generating suggested prompts using LLM...")
        response = llm.complete(prompt)
        suggestions_text = response.text.strip()

        # Parse the response (split by newline, remove empty lines)
        suggested_prompts = [line.strip() for line in suggestions_text.split('\n') if line.strip()]

        # Validate the output
        if len(suggested_prompts) == SUGGESTED_PROMPT_COUNT and all(suggested_prompts):
            print(f"LLM generated prompts: {suggested_prompts}")
            return suggested_prompts
        else:
            print(f"Warning: LLM generated unexpected output for suggestions: '{suggestions_text}'. Falling back to defaults.")
            return DEFAULT_PROMPTS

    except Exception as e:
        print(f"Error generating suggested prompts with LLM: {e}. Falling back to defaults.")
        return DEFAULT_PROMPTS


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    try:
        pass

    except Exception as e:
        print(f"Error during agent testing: {e}")
        import traceback
        traceback.print_exc()
