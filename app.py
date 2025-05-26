import streamlit as st
import os
import json
import re
from typing import Any, Optional, Dict, List
from llama_index.core.llms import ChatMessage, MessageRole
import stui
from agent import create_unified_agent, generate_suggested_prompts, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv
from streamlit_oauth import OAuth2Component
import user_data_manager # New import for user data persistence
import requests # Added for fetching user info from Google OAuth

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


# --- Constants and Configuration ---
AGENT_SESSION_KEY = "esi_unified_agent" # Key for storing unified agent
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---" # Used by stui.py for display
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---" # Used by stui.py for display

# OAuth Configuration
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"

# Redirect URI for local development. Adjust for deployment.
# Ensure this matches the authorized redirect URI in your Google Cloud Console.
REDIRECT_URI = "http://localhost:8501" 

# Scopes for Google user info
SCOPE = "https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile openid"


# --- Session State Initialization ---
def init_session_state_for_app():
    """Initializes core session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Will be loaded/initialized after login

    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = [] # Will be generated after login/initial greeting

    if "user_info" not in st.session_state:
        st.session_state.user_info = None # Stores logged-in user's Google info

    if "current_discussion_id" not in st.session_state:
        st.session_state.current_discussion_id = None # ID of the currently active discussion

    if "discussion_title" not in st.session_state:
        st.session_state.discussion_title = "New Discussion" # Title of the current discussion

    if "all_discussions" not in st.session_state:
        st.session_state.all_discussions = [] # List of all discussions for the logged-in user

    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7 # Default LLM temperature


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
    if not st.session_state.user_info:
        st.warning("Please log in to start a discussion.")
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
    user_id = st.session_state.user_info['id']
    new_discussion_meta = user_data_manager.create_new_discussion(user_id)
    st.session_state.current_discussion_id = new_discussion_meta["id"]
    st.session_state.discussion_title = new_discussion_meta["title"]
    st.session_state.messages = [{"role": "assistant", "content": stui.get_greeting_message()}]
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    _refresh_discussion_list()
    print(f"New discussion created and set as current: {st.session_state.discussion_title} ({st.session_state.current_discussion_id})")
    st.rerun()

def _load_discussion_session(discussion_id: str):
    """Loads an existing discussion and sets it as current."""
    user_id = st.session_state.user_info['id']
    discussion_data = user_data_manager.load_discussion(user_id, discussion_id)
    if discussion_data:
        st.session_state.current_discussion_id = discussion_data["id"]
        st.session_state.discussion_title = discussion_data.get("title", "Untitled Discussion")
        st.session_state.messages = discussion_data.get("messages", [])
        if not st.session_state.messages: # If loaded discussion is empty, add greeting
             st.session_state.messages = [{"role": "assistant", "content": stui.get_greeting_message()}]
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
        print(f"Loaded discussion: {st.session_state.discussion_title} ({st.session_state.current_discussion_id})")
    else:
        st.error("Failed to load discussion.")
        print(f"Failed to load discussion with ID: {discussion_id}")
    st.rerun()

def _save_current_discussion():
    """Saves the current discussion to persistent storage."""
    if st.session_state.user_info and st.session_state.current_discussion_id:
        user_id = st.session_state.user_info['id']
        user_data_manager.save_discussion(
            user_id,
            st.session_state.current_discussion_id,
            st.session_state.discussion_title,
            st.session_state.messages
        )
        _refresh_discussion_list() # Refresh list to update 'updated_at' or title if changed
    else:
        print("Cannot save: No user logged in or no current discussion.")

def _delete_current_discussion():
    """Deletes the currently active discussion."""
    if st.session_state.user_info and st.session_state.current_discussion_id:
        user_id = st.session_state.user_info['id']
        if user_data_manager.delete_discussion(user_id, st.session_state.current_discussion_id):
            st.success(f"Discussion '{st.session_state.discussion_title}' deleted.")
            st.session_state.current_discussion_id = None
            st.session_state.discussion_title = "New Discussion"
            st.session_state.messages = [] # Clear messages
            _refresh_discussion_list()
            st.rerun()
        else:
            st.error("Failed to delete discussion.")
    else:
        st.warning("No discussion selected to delete.")

def _refresh_discussion_list():
    """Refreshes the list of discussions for the current user."""
    if st.session_state.user_info:
        st.session_state.all_discussions = user_data_manager.list_discussions(st.session_state.user_info['id'])
    else:
        st.session_state.all_discussions = []

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
        new_greeting = generate_llm_greeting()
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

    # --- Google OAuth Login/Logout ---
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)

    with st.sidebar:
        st.header("User Account")
        if st.session_state.user_info:
            user_email = st.session_state.user_info.get('email', 'N/A')
            user_name = st.session_state.user_info.get('name', 'User')
            st.write(f"Logged in as: **{user_name}** ({user_email})")
            if st.button("Logout", key="logout_button"):
                oauth2.revoke_token()
                st.session_state.user_info = None
                st.session_state.messages = []
                st.session_state.current_discussion_id = None
                st.session_state.discussion_title = "New Discussion"
                st.session_state.all_discussions = []
                st.session_state.suggested_prompts = []
                st.rerun()
        else:
            result = oauth2.authorize_redirect(
                name="Login with Google",
                redirect_uri=REDIRECT_URI,
                scope=SCOPE,
                height=600,
                width=500,
                key="google_login_button"
            )
            if result and result.get("token"):
                # Fetch user info
                headers = {"Authorization": f"Bearer {result.get('token')['access_token']}"}
                user_info_response = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
                if user_info_response.status_code == 200:
                    st.session_state.user_info = user_info_response.json()
                    print(f"User logged in: {st.session_state.user_info.get('email')}")
                    # After successful login, load user's discussions
                    _refresh_discussion_list()
                    # If no discussions, create a new one
                    if not st.session_state.all_discussions:
                        _create_new_discussion_session()
                    else: # Load the most recent one
                        _load_discussion_session(st.session_state.all_discussions[0]['id'])
                    st.rerun()
                else:
                    st.error("Failed to fetch user info from Google.")
                    print(f"Failed to fetch user info: {user_info_response.status_code} - {user_info_response.text}")
            elif result and result.get("error"):
                st.error(f"Login failed: {result.get('error')}")
                print(f"OAuth error: {result.get('error')}")

    # Only show chat interface if logged in
    if st.session_state.user_info:
        # Handle regeneration request if flag is set
        if st.session_state.get("do_regenerate", False):
            handle_regeneration_request()

        # Create the rest of the interface using stui (displays chat history, sidebar, etc.)
        stui.create_interface()

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
    else:
        st.info("Please log in with your Google account to start using ESI and access persistent memory features.")
        st.image("https://www.uea.ac.uk/documents/20145/1000000/UEA_Logo_RGB_White_Background.jpg", width=200)
        st.markdown("---")
        st.markdown("Welcome to ESI, your AI assistant for dissertation support. Log in to save your conversations!")


if __name__ == "__main__":
    # Display a warning if environment variables are missing
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")
    if not CLIENT_ID or not CLIENT_SECRET:
        st.warning("⚠️ GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET environment variables not set. Google login will not work.")

    main()
