import streamlit as st
import os
import json
import uuid # For generating unique user IDs
from typing import Any, Optional, Dict, List
from llama_index.core.llms import ChatMessage, MessageRole
import stui

from agent import create_unified_agent, generate_suggested_prompts, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv

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

# IMPORTANT: This block waits for cookies to be ready.
# The `st.stop()` here is correct for `streamlit_cookies_manager`.
if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.info("Loading user session...") # Provide feedback to the user
    st.stop() # Reverted to st.stop() to fix SyntaxError


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

    if "do_regenerate" not in st.session_state:
        st.session_state.do_regenerate = False

    if "should_generate_prompts" not in st.session_state: # New flag for controlled prompt generation
        st.session_state.should_generate_prompts = False

    if "editing_list_discussion_id" not in st.session_state: # New flag for inline editing in list
        st.session_state.editing_list_discussion_id = None

    if "user_id_initialized" not in st.session_state: # New flag to control initial setup
        st.session_state.user_id_initialized = False

    if "discussion_setup_done" not in st.session_state: # New flag for one-time discussion setup
        st.session_state.discussion_setup_done = False


# --- Helper Function for History Formatting ---
def format_chat_history(streamlit_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        # Content is now expected to be a string for all messages after reversion
        content = msg["content"]
        history.append(ChatMessage(role=role, content=content))

    return history


# --- Agent Interaction ---
def get_agent_response(query: str, chat_history: list[ChatMessage]) -> str:
    """
    Get a response from the agent stored in the session state using the chat method,
    explicitly passing the conversation history. Returns a string response.
    """
    if AGENT_SESSION_KEY not in st.session_state or st.session_state[AGENT_SESSION_KEY] is None:
        return "Error: Agent not initialized. Please refresh the page."

    agent_runner = st.session_state[AGENT_SESSION_KEY]
    original_response_text: str

    try:
        current_temperature = st.session_state.get("llm_temperature", 0.7)

        # Accessing the LLM instance within AgentRunner -> FunctionCallingAgentWorker -> LLM
        if hasattr(agent_runner, '_agent_worker') and hasattr(agent_runner._agent_worker, '_llm'):
            actual_llm_instance = agent_runner._agent_worker._llm
            if hasattr(actual_llm_instance, 'temperature'):
                actual_llm_instance.temperature = current_temperature
                print(f"Set agent LLM temperature to: {current_temperature}")
            else:
                print(f"Warning: Agent LLM object of type {type(actual_llm_instance)} does not have a 'temperature' attribute.")
        else:
            print("Warning: Could not access LLM object within the agent to set temperature.")

        with st.spinner("ESI is thinking..."): # Simplified spinner message
            response_obj = agent_runner.chat(query, chat_history=chat_history)
                                                                 # ^ Changed from formatted_history

        response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)

        print(f"Unified agent final response text for UI: \n{response_text[:500]}...")
        return response_text

    except Exception as e:
        print(f"Error getting unified agent response: {e}")
        return f"I apologize, but I encountered an error while processing your request with the main agent. Please try again. Technical details: {str(e)}"


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
        st.session_state.editing_list_discussion_id = None # Exit edit mode if a prompt is used
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None
    elif chat_input_value:
        st.session_state.editing_list_discussion_id = None # Exit edit mode if chat input is used
        prompt_to_process = chat_input_value

    if prompt_to_process:
        # If no discussion is active, create a new one (this should ideally be handled by main loop)
        # This check is a fallback, but the main loop should ensure current_discussion_id is set.
        if not st.session_state.current_discussion_id:
            _create_new_discussion_session() # This will set the current discussion and messages
            # No rerun here, as the subsequent append/get_agent_response will trigger it.

        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        formatted_history = format_chat_history(st.session_state.messages)
        response_text_string = get_agent_response(prompt_to_process, chat_history=formatted_history)
        st.session_state.messages.append({"role": "assistant", "content": response_text_string})


        # Save the updated discussion after each turn
        _save_current_discussion()
        st.session_state.should_generate_prompts = True # Set flag to generate new prompts
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
    
    # Initialize messages with a greeting
    greeting_text = generate_llm_greeting() # Direct call to agent.py
    st.session_state.messages = [{"role": "assistant", "content": greeting_text}]

    # Directly generate prompts:
    print("Generating initial suggested prompts directly within _create_new_discussion_session...")
    # Ensure generate_suggested_prompts is available/imported if not already.
    # It is imported from agent.py at the top of app.py
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.session_state.should_generate_prompts = False # Reset flag immediately

    st.session_state.editing_list_discussion_id = None # Exit any inline editing mode

    _refresh_discussion_list()
    print(f"New discussion created and set as current: {st.session_state.current_discussion_title} ({st.session_state.current_discussion_id})")
    # Removed st.rerun() - button click will trigger natural rerun

def _load_discussion_session(discussion_id: str):
    """Loads an existing discussion and sets it as current."""
    user_id = st.session_state.user_id
    discussion_data = user_data_manager.load_discussion(user_id, discussion_id)
    if discussion_data:
        st.session_state.current_discussion_id = discussion_data["id"]
        st.session_state.current_discussion_title = discussion_data.get("title", "Untitled Discussion") # Standardized name
        st.session_state.messages = discussion_data.get("messages", [])
        
        if not st.session_state.messages: # If loaded discussion is empty, add greeting
             greeting_text = generate_llm_greeting() # Direct call to agent.py
             st.session_state.messages = [{"role": "assistant", "content": greeting_text}]

        
        st.session_state.should_generate_prompts = True # Set flag to generate new prompts
        st.session_state.editing_list_discussion_id = None # Exit any inline editing mode
        print(f"Loaded discussion: {st.session_state.current_discussion_title} ({st.session_state.current_discussion_id})")
    else:
        st.error("Failed to load discussion.")
        print(f"Failed to load discussion with ID: {discussion_id}")
    _refresh_discussion_list() # Refresh list to ensure current discussion is highlighted
    # Removed st.rerun() here. State changes should trigger re-render.

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

def _delete_current_discussion(discussion_id_to_delete: Optional[str] = None):
    """Deletes the specified discussion or the currently active one if no ID is provided."""
    target_discussion_id = discussion_id_to_delete if discussion_id_to_delete else st.session_state.current_discussion_id

    if st.session_state.user_id and target_discussion_id:
        # Find the title for the success message before deleting
        deleted_title = "Unknown Discussion"
        for disc in st.session_state.discussion_list:
            if disc['id'] == target_discussion_id:
                deleted_title = disc['title']
                break

        if user_data_manager.delete_discussion(st.session_state.user_id, target_discussion_id):
            st.success(f"Discussion '{deleted_title}' deleted.")
            
            # If the deleted discussion was the current one, reset current discussion state
            if target_discussion_id == st.session_state.current_discussion_id:
                st.session_state.current_discussion_id = None
                st.session_state.current_discussion_title = "New Discussion"
                st.session_state.messages = []
            
            _refresh_discussion_list() # Always refresh the list after deletion
            st.session_state.editing_list_discussion_id = None # Exit any inline editing mode
            # Removed st.rerun() here. The state changes should trigger a re-render.
            # The main loop will handle creating a new discussion if current_discussion_id is None.
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
        # If 'timestamp' is not present, .get('timestamp', '') will return an an empty string,
        # which will still allow sorting without crashing, though the order might not be ideal.
        st.session_state.discussion_list = sorted(discussions, key=lambda x: x.get('timestamp', ''), reverse=True)
        print(f"Listed {len(st.session_state.discussion_list)} discussions for user {st.session_state.user_id}.")
    else:
        st.session_state.discussion_list = []

def _update_listed_discussion_title(discussion_id: str):
    """Updates the title of a specific discussion in the list and saves it."""
    new_title = st.session_state[f"edit_title_input_{discussion_id}"] # Get value from the specific text_input
    
    # Find the discussion in the list and update its title
    for i, disc in enumerate(st.session_state.discussion_list):
        if disc['id'] == discussion_id:
            st.session_state.discussion_list[i]['title'] = new_title
            # If it's the current discussion, also update its title
            if discussion_id == st.session_state.current_discussion_id:
                st.session_state.current_discussion_title = new_title
            break
    
    # Save the updated discussion to disk
    # Need to load the full discussion data first, update title, then save
    discussion_data = user_data_manager.load_discussion(st.session_state.user_id, discussion_id)
    if discussion_data:
        discussion_data['title'] = new_title
        user_data_manager.save_discussion(
            st.session_state.user_id,
            discussion_id,
            new_title, # Pass the new title explicitly
            discussion_data['messages'] # Pass existing messages
        )
        _refresh_discussion_list() # Refresh to ensure list is consistent
    else:
        print(f"Warning: Could not find discussion {discussion_id} to update title.")

def _get_discussion_markdown(discussion_id: str) -> str:
    """Loads a specific discussion and converts its chat history to a Markdown string."""
    user_id = st.session_state.user_id
    discussion_data = user_data_manager.load_discussion(user_id, discussion_id)
    
    if not discussion_data:
        return "Discussion not found."

    markdown_content = f"# Discussion: {discussion_data.get('title', 'Untitled Discussion')}\n\n"
    for message in discussion_data.get('messages', []):
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"## {role}\n{content}\n\n"

    return markdown_content


# --- UI Callbacks ---
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
        new_greeting_text = generate_llm_greeting() # Direct call to agent.py
        st.session_state.messages[0]['content'] = new_greeting_text

        _save_current_discussion() # Save the regenerated greeting
        st.session_state.should_generate_prompts = True # Set flag to generate new prompts
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

    response_text_string = get_agent_response(prompt_to_regenerate, chat_history=formatted_history_for_regen)
    st.session_state.messages.append({"role": "assistant", "content": response_text_string})
    _save_current_discussion() # Save the regenerated response
    st.session_state.should_generate_prompts = True # Set flag to generate new prompts
    st.rerun()


# Expose these functions to st.session_state so stui.py can call them
# This block MUST be before stui.create_interface() call
st.session_state._create_new_discussion_session = _create_new_discussion_session
st.session_state._load_discussion_session = _load_discussion_session
st.session_state._save_current_discussion = _save_current_discussion
st.session_state._delete_current_discussion = _delete_current_discussion
st.session_state._refresh_discussion_list = _refresh_discussion_list
st.session_state.handle_regeneration_request = handle_regeneration_request # Expose for stui.py
st.session_state._update_listed_discussion_title = _update_listed_discussion_title # Expose new function
st.session_state._get_discussion_markdown = _get_discussion_markdown # Expose new function


def main():
    """Main function to run the Streamlit app."""
    init_session_state_for_app() # Initializes 'user_id_initialized = False', 'discussion_setup_done = False'

    # This check for cookies.ready() is from the original code and should be kept.
    # It uses st.stop() correctly for streamlit_cookies_manager.
    if not cookies.ready():
        st.info("Loading user session...")
        st.stop()

    # Block 1: User ID Initialization
    # Goal: Ensure st.session_state.user_id is set and st.session_state.user_id_initialized is True.
    # This block should run until user_id_initialized is True.
    if not st.session_state.user_id_initialized:
        print("Attempting to initialize user_id...")
        user_id_from_cookie = cookies.get("user_id")

        if user_id_from_cookie:
            st.session_state.user_id = user_id_from_cookie
            st.session_state.user_id_initialized = True
            print(f"User ID {user_id_from_cookie} loaded from cookie. user_id_initialized=True.")
            # No st.rerun() needed here, proceed to Block 2 if conditions met
        else:
            new_user_id = str(uuid.uuid4())
            st.session_state.user_id = new_user_id
            # Set initialized flag BEFORE save, to ensure it's set if the rerun reads it.
            st.session_state.user_id_initialized = True 
            cookies["user_id"] = new_user_id # Set cookie
            print(f"New user ID {new_user_id} generated. user_id_initialized=True. Saving to cookie...")
            cookies.save() # This triggers a rerun by the cookie manager
            print("Cookie saved. Halting current script execution for Streamlit rerun.")
            st.stop() # Halt current run, let the save-induced rerun take over.
    
    # Block 2: Agent and One-Time Discussion Setup
    # This block runs only if user_id is stable AND discussion setup hasn't occurred yet.
    if st.session_state.user_id_initialized and not st.session_state.discussion_setup_done:
        print("User ID initialized. Performing one-time Agent and Discussion setup...")
        
        # Initialize LLM settings and Unified Agent
        initialize_agent_settings() # From agent.py
        if AGENT_SESSION_KEY not in st.session_state or st.session_state[AGENT_SESSION_KEY] is None:
            st.session_state[AGENT_SESSION_KEY] = create_unified_agent() # From agent.py
            print("Unified agent created and stored in session state.")
        else:
            print("Unified agent already exists in session state.")

        _refresh_discussion_list() # Populate st.session_state.discussion_list

        if not st.session_state.discussion_list: # No discussions at all for this user
            print("No existing discussions found. Creating a new initial discussion.")
            _create_new_discussion_session() # This calls generate_suggested_prompts directly
        elif not st.session_state.current_discussion_id: # Has discussions, but none is set as current
            print("Existing discussions found, loading the most recent one as current.")
            if st.session_state.discussion_list: # Should be sorted by _refresh_discussion_list
                 _load_discussion_session(st.session_state.discussion_list[0]['id']) 
                 # _load_discussion_session sets should_generate_prompts=True
            else: 
                 print("Warning: Discussion list exists but is empty after refresh. Creating new discussion.")
                 _create_new_discussion_session()
        else:
            # Current discussion ID already exists. We assume it's correctly loaded.
            # If messages were empty for some reason, _load_discussion_session handles adding a greeting.
            # And _load_discussion_session also sets should_generate_prompts = True.
            # To be safe, and ensure prompts are generated if this is the first load after setup,
            # we can call _load_discussion_session here. It should be idempotent.
            print(f"Current discussion ID {st.session_state.current_discussion_id} already set. Ensuring it's fully loaded and prompts are triggered.")
            _load_discussion_session(st.session_state.current_discussion_id)


        st.session_state.discussion_setup_done = True
        print("One-time Agent and Discussion setup complete. discussion_setup_done=True.")
        # Important: Rerun to ensure the UI reflects the loaded/new discussion and suggested prompts properly.
        # The _load_discussion_session or _create_new_discussion_session would have set messages
        # and potentially set should_generate_prompts=True (or directly generated them in create_new).
        # This rerun ensures the rest of the main() function (Block 3 and UI rendering)
        # operates with the fully initialized state.
        st.rerun() 

    # Block 3: Fallback if no current discussion is active AFTER initial setup
    # (e.g., user deleted the active discussion, or an error occurred loading previously)
    # This should only execute if user_id is initialized AND one-time setup is done.
    if st.session_state.user_id_initialized and \
       st.session_state.discussion_setup_done and \
       not st.session_state.current_discussion_id:
        print("No current discussion active after initial setup (e.g., was deleted or failed to load). Creating a new one.")
        _create_new_discussion_session() # This will set up a new discussion and messages
        # _create_new_discussion_session now directly generates prompts.
        st.rerun() # Rerun to reflect the new discussion state and its prompts.

    # --- Main Chat Interface Logic (Original structure follows) ---

    # Handle regeneration request if flag is set
    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request() # This function itself calls st.rerun()

    # Generate suggested prompts if the flag is set (e.g., after a user message, or by _load_discussion_session)
    # This block is now more controlled due to earlier setup.
    if st.session_state.should_generate_prompts:
        if not st.session_state.get("suggested_prompts") or not st.session_state.messages:
            # Only generate if prompts are missing, or if messages list is empty (should not happen if greeting is always there)
            # The main trigger for regeneration of prompts after user input is in handle_user_input.
            # _load_discussion_session and _create_new_discussion_session also manage this.
            print("Generating suggested prompts based on current messages (should_generate_prompts=True)...")
            st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
        else:
            print("Suggested prompts already exist or messages are empty; skipping generation in main loop's check.")
        st.session_state.should_generate_prompts = False # Reset the flag

    # Create the rest of the interface using stui (displays chat history, sidebar, etc.)
    # This needs a valid discussion (messages, title) to be set up.
    if st.session_state.current_discussion_id and st.session_state.user_id_initialized and st.session_state.discussion_setup_done:
        stui.create_interface(
            DOWNLOAD_MARKER=DOWNLOAD_MARKER,
            RAG_SOURCE_MARKER_PREFIX=RAG_SOURCE_MARKER_PREFIX
        )

        # Render the chat input box at the bottom, capture its value
        chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")

        # Handle user input (either from chat box or a clicked suggested prompt button)
        handle_user_input(chat_input_value) # This function calls st.rerun() after processing
    elif not st.session_state.user_id_initialized:
        st.warning("User session not yet initialized. Please wait or refresh.")
    elif not st.session_state.discussion_setup_done:
        st.info("Setting up your discussion environment. Please wait...")
        # Potentially add a st.rerun() here if it seems stuck, but the logic above should handle it.
    else: # Should be caught by Block 3, but as a fallback:
        st.error("Something went wrong with discussion setup. Please try refreshing the page.")
        print("Error state: user_id_initialized={}, discussion_setup_done={}, current_discussion_id={}".format(
            st.session_state.user_id_initialized, 
            st.session_state.discussion_setup_done, 
            st.session_state.current_discussion_id
        ))


    # Display a warning if Google API Key is missing
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")


if __name__ == "__main__":
    main()
