import streamlit as st
import os
import json
import re
import uuid # For generating unique user IDs
from typing import Any, Optional, Dict, List
from llama_index.core.llms import ChatMessage, MessageRole
import stui
from agent import create_unified_agent, generate_suggested_prompts, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv

# Moved import to after set_page_config to ensure it's not the cause of the "first command" error
# from streamlit_cookies_manager import CookieManager # For cookie-based persistence
import user_data_manager # New import for user data persistence

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

# Configure page settings - MUST be the first Streamlit command
st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import EncryptedCookieManager after set_page_config
from streamlit_cookies_manager import EncryptedCookieManager # For cookie-based persistence
import os # For environment variables for cookie password

# --- Initialize Cookie Manager (should be on top) ---
cookies = EncryptedCookieManager(
    prefix="esi_app_user/", # Example prefix
    # You should really setup a long COOKIES_PASSWORD secret if you're running on Streamlit Cloud.
    password=os.environ.get("COOKIES_PASSWORD", "a_secure_default_password_for_esi_app"),
)

if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.stop()

# --- Constants and Configuration ---
AGENT_SESSION_KEY = "esi_unified_agent" # Key for storing unified agent
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---" # Used by stui.py for display
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---" # Used by stui.py for display


# --- Session State Initialization ---
def init_session_state_for_app():
    """Initializes core session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Will be loaded/initialized after user ID is set

    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = [] # Will be generated after initial greeting

    if "user_id" not in st.session_state: # Changed from user_info
        st.session_state.user_id = None # Stores logged-in user's ID

    if "current_discussion_id" not in st.session_state:
        st.session_state.current_discussion_id = None # ID of the currently active discussion

    if "current_discussion_title" not in st.session_state: # Standardized name
        st.session_state.current_discussion_title = "New Discussion" # Title of the current discussion

    if "discussion_list" not in st.session_state: # Standardized name
        st.session_state.discussion_list = [] # List of all discussions for the logged-in user

    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7 # Default LLM temperature

    if "next_research_idea_number" not in st.session_state:
        st.session_state.next_research_idea_number = 1


# --- Agent Initialization ---
def initialize_agent():
    """Initializes the unified agent and stores it in session state."""
    if AGENT_SESSION_KEY not in st.session_state:
        print("Initializing unified agent for new session (LLM settings should already be done)...")
        try:
            st.session_state[AGENT_SESSION_KEY] = create_unified_agent()
            print("Unified agent object initialized successfully.")
        except Exception as e:
            print(f"Error initializing unified agent: {e}")
            st.error(f"Failed to initialize the AI agent. Please check configurations. Error: {e}")
            st.stop() # Stop execution if agent fails to initialize

# --- Helper Function for History Formatting ---
def format_chat_history(streamlit_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history


# --- Agent Interaction ---
def get_agent_response(query: str, chat_history: list[ChatMessage]) -> str:
    """
    Get a response from the agent stored in the session state using the chat method,
    explicitly passing the conversation history.
    """
    if AGENT_SESSION_KEY not in st.session_state or st.session_state[AGENT_SESSION_KEY] is None:
        return "Error: Agent not initialized. Please refresh the page."

    agent = st.session_state[AGENT_SESSION_KEY]

    try:
        current_temperature = st.session_state.get("llm_temperature", 0.7)

        if hasattr(agent, '_agent_worker') and hasattr(agent._agent_worker, '_llm'):
            actual_llm_instance = agent._agent_worker._llm
            if hasattr(actual_llm_instance, 'temperature'):
                actual_llm_instance.temperature = current_temperature
                print(f"Set LLM temperature to: {current_temperature}")
            else:
                print(f"Warning: LLM object of type {type(actual_llm_instance)} does not have a 'temperature' attribute.")
        else:
            print("Warning: Could not access LLM object within the agent to set temperature. Agent or worker structure might have changed (_agent_worker or _agent_worker._llm not found).")

        with st.spinner("ESI is thinking..."):
            response = agent.chat(query, chat_history=chat_history)

        response_text = response.response if hasattr(response, 'response') else str(response)

        print(f"Unified agent final response text for UI: \n{response_text[:500]}...")
        return response_text

    except Exception as e:
        print(f"Error getting unified agent response: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"

def handle_user_input(chat_input_value: str | None):
    """
    Process user input (either from chat box or suggested prompt)
    and update chat with AI response.
    """
    # User ID should always be present after main() runs
    if not st.session_state.user_id:
        st.error("User not identified. Please refresh the page.")
        return

    prompt_to_process = None

    if hasattr(st.session_state, 'prompt_to_use') and st.session_state.prompt_to_use:
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None
    elif chat_input_value:
        prompt_to_process = chat_input_value

    if prompt_to_process:
        # If no discussion is active, create a new one
        if not st.session_state.current_discussion_id:
            _create_new_discussion_session()

        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        formatted_history = format_chat_history(st.session_state.messages)
        response_text = get_agent_response(prompt_to_process, chat_history=formatted_history)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Save the updated discussion after each turn
        _save_current_discussion()
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
        st.rerun()

# --- Discussion Management Functions ---
def _create_new_discussion_session():
    """Creates a new discussion and sets it as current."""
    user_id = st.session_state.user_id
    new_title = f"Research idea {st.session_state.next_research_idea_number}"
    st.session_state.next_research_idea_number += 1

    new_discussion_meta = user_data_manager.create_new_discussion(user_id, new_title)
    st.session_state.current_discussion_id = new_discussion_meta["id"]
    st.session_state.current_discussion_title = new_discussion_meta["title"] # Standardized name
    
    # Initialize messages with a greeting and generate suggested prompts
    st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}] # Direct call to agent.py
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)

    _refresh_discussion_list()
    print(f"New discussion created and set as current: {st.session_state.current_discussion_title} ({st.session_state.current_discussion_id})")
    st.rerun()

def _load_discussion_session(discussion_id: str):
    """Loads an existing discussion and sets it as current."""
    user_id = st.session_state.user_id
    discussion_data = user_data_manager.load_discussion(user_id, discussion_id)
    if discussion_data:
        st.session_state.current_discussion_id = discussion_data["id"]
        st.session_state.current_discussion_title = discussion_data.get("title", "Untitled Discussion") # Standardized name
        st.session_state.messages = discussion_data.get("messages", [])
        
        if not st.session_state.messages: # If loaded discussion is empty, add greeting
             st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}] # Direct call to agent.py
        
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
        print(f"Loaded discussion: {st.session_state.current_discussion_title} ({st.session_state.current_discussion_id})")
    else:
        st.error("Failed to load discussion.")
        print(f"Failed to load discussion with ID: {discussion_id}")
        _create_new_discussion_session() # Fallback to new discussion if load fails
    st.rerun()

def _save_current_discussion():
    """Saves the current discussion to persistent storage."""
    if st.session_state.user_id and st.session_state.current_discussion_id:
        user_id = st.session_state.user_id
        user_data_manager.save_discussion(
            user_id,
            st.session_state.current_discussion_id,
            st.session_state.current_discussion_title, # Standardized name
            st.session_state.messages
        )
        _refresh_discussion_list() # Refresh list to update 'updated_at' or title if changed
    else:
        print("Cannot save: No user identified or no current discussion.")

def _delete_current_discussion():
    """Deletes the currently active discussion."""
    if st.session_state.user_id and st.session_state.current_discussion_id:
        user_id = st.session_state.user_id
        if user_data_manager.delete_discussion(user_id, st.session_state.current_discussion_id):
            st.success(f"Discussion '{st.session_state.current_discussion_title}' deleted.") # Standardized name
            st.session_state.current_discussion_id = None
            st.session_state.current_discussion_title = "New Discussion" # Standardized name
            st.session_state.messages = [] # Clear messages
            _refresh_discussion_list()
            st.rerun()
        else:
            st.error("Failed to delete discussion.")
    else:
        st.warning("No discussion selected to delete.")

def _refresh_discussion_list():
    """Refreshes the list of discussions for the current user, sorted by most recent."""
    if st.session_state.user_id:
        discussions = user_data_manager.list_discussions(st.session_state.user_id)
        # Sort by 'timestamp' in descending order (most recent first)
        # Assuming each discussion dictionary has a 'timestamp' key.
        # If 'timestamp' is not present, .get('timestamp', '') will return an empty string,
        # which will still allow sorting without crashing, though the order might not be ideal.
        st.session_state.discussion_list = sorted(discussions, key=lambda x: x.get('timestamp', ''), reverse=True)
    else:
        st.session_state.discussion_list = []

# Expose these functions to st.session_state so stui.py can call them
st.session_state._create_new_discussion_session = _create_new_discussion_session
st.session_state._load_discussion_session = _load_discussion_session
st.session_state._save_current_discussion = _save_current_discussion
st.session_state._delete_current_discussion = _delete_current_discussion
st.session_state._refresh_discussion_list = _refresh_discussion_list


# --- UI Callbacks ---
def set_selected_prompt_from_dropdown():
    """Callback function to set the selected prompt from the dropdown."""
    prompt = st.session_state.get("selected_prompt_dropdown", "")
    if prompt:
        st.session_state.prompt_to_use = prompt

def reset_chat_callback():
    """Resets the current chat history and suggested prompts to their initial state."""
    print("Resetting current chat...")
    # This will create a new discussion if one isn't active, or clear the current one
    _create_new_discussion_session()
    st.rerun()

def handle_regeneration_request():
    """Handles the request to regenerate the last assistant response."""
    if not st.session_state.get("do_regenerate", False):
        return

    st.session_state.do_regenerate = False # Consume the flag

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'assistant':
        print("Warning: Regeneration called but last message is not from assistant or no messages exist.")
        st.rerun()
        return

    # Case 1: Regenerating the initial greeting
    if len(st.session_state.messages) == 1:
        print("Regenerating initial greeting...")
        new_greeting = generate_llm_greeting() # Direct call to agent.py
        st.session_state.messages[0]['content'] = new_greeting
        _save_current_discussion() # Save the regenerated greeting
        st.rerun()
        return

    # Case 2: Regenerating a response to a user query
    st.session_state.messages.pop() # Remove last assistant message

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'user':
        print("Warning: Cannot regenerate, no preceding user query found after popping assistant message.")
        st.rerun()
        return

    print("Regenerating last assistant response to user query...")
    prompt_to_regenerate = st.session_state.messages[-1]['content']
    formatted_history_for_regen = format_chat_history(st.session_state.messages)

    response_text = get_agent_response(prompt_to_regenerate, chat_history=formatted_history_for_regen)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    _save_current_discussion() # Save the regenerated response
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()


def main():
    """Main function to run the Streamlit app."""
    init_session_state_for_app()

    # --- Initialize LLM Settings FIRST ---
    try:
        print("Initializing LLM settings...")
        initialize_agent_settings()
        print("LLM settings initialized.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize LLM settings. {e}")
        st.stop()
    # --- End LLM Settings Initialization ---

    # Initialize agent (stores it in session state)
    initialize_agent()

    # --- User Identification (Cookie-based) ---
    if "user_id" not in st.session_state or st.session_state.user_id is None: # Changed from user_info
        user_id = None
        try:
            user_id = cookies["user_id"] # Use dictionary-style access
            if user_id:
                print(f"Loaded user ID from cookie: {user_id}")
            else: # Cookie exists but is empty
                print("User ID cookie was empty, generating a new one.")
                user_id = str(uuid.uuid4())
                cookies["user_id"] = user_id
                cookies.save() # Explicitly save
                print(f"Generated new user ID and set cookie: {user_id}")
        except KeyError:
            print("User ID cookie not found, generating a new one.")
            user_id = str(uuid.uuid4())
            cookies["user_id"] = user_id # Set using dictionary-style access
            cookies.save() # Explicitly save
            print(f"Generated new user ID and set cookie: {user_id}")
        except Exception as e:
            st.error(f"An unexpected error occurred with cookies: {e}. Please try clearing your browser cookies for this site.")
            # Fallback to a temporary session-based ID if cookies fail catastrophically
            if 'user_id_temp' not in st.session_state:
                st.session_state.user_id_temp = str(uuid.uuid4())
            user_id = st.session_state.user_id_temp
            print(f"Fell back to temporary user ID: {user_id}")

        st.session_state.user_id = user_id # Standardized name
        
        # After establishing user_id, load user's discussions
        _refresh_discussion_list()
        # If no discussions, create a new one
        if not st.session_state.discussion_list: # Standardized name
            _create_new_discussion_session()
        else: # Load the most recent one
            _load_discussion_session(st.session_state.discussion_list[0]['id']) # Standardized name
        st.rerun() # Rerun to apply user_info and load discussion

    # --- Main Chat Interface ---
    # Handle regeneration request if flag is set
    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request()

    # Create the rest of the interface using stui (displays chat history, sidebar, etc.)
    stui.create_interface(
        DOWNLOAD_MARKER=DOWNLOAD_MARKER,
        RAG_SOURCE_MARKER_PREFIX=RAG_SOURCE_MARKER_PREFIX
    )

    # Display suggested prompts as a dropdown below the chat history
    st.selectbox(
        "Select a suggested prompt:",
        options=[""] + st.session_state.suggested_prompts,
        key="selected_prompt_dropdown",
        placeholder="Select a suggested prompt...",
        on_change=set_selected_prompt_from_dropdown
    )

    # Render the chat input box at the bottom, capture its value
    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")

    # Handle user input (either from chat box or a clicked suggested prompt button)
    handle_user_input(chat_input_value)

    # Display a warning if Google API Key is missing (moved here)
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")


if __name__ == "__main__":
    main()
