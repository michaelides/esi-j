import streamlit as st
import os
import time
import json
import re # Import regex module for parsing code blocks and markers
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole # Import necessary types
import stui
# Import initialize_settings and alias it, and the new orchestrator agent creator
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings
# UI_ACCESSIBLE_WORKSPACE is now primarily managed within tools.py and coder agent
# from tools import UI_ACCESSIBLE_WORKSPACE
# PERSISTENT_SESSIONS_DIR = "./persistent_sessions" # Removed
from dotenv import load_dotenv
# import shutil # No longer needed here, file moving logic removed

# Load environment variables
load_dotenv()

# Configure page settings - MUST be the first Streamlit command
st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor", # Consistent title
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Constants and Configuration ---
DB_PATH = os.getenv("DB_PATH", "./ragdb/chromadb")
AGENT_SESSION_KEY = "esi_orchestrator_agent" # Key for storing orchestrator agent
# CODE_INTERPRETER_TOOL_NAME = "code_interpreter" # Managed by Coder Agent
# RAG_TOOL_NAME = "rag_dissertation_retriever" # Managed by RAG Agent, tool name for orchestrator is "rag_expert"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---" # Used by stui.py for display
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---" # Used by stui.py for display

# --- Agent Initialization ---
def initialize_agent():
    """Initializes the orchestrator agent and stores it in session state."""
    if AGENT_SESSION_KEY not in st.session_state:
        print("Initializing orchestrator agent for new session (LLM settings should already be done)...")
        try:
            # create_orchestrator_agent uses the globally initialized Settings.llm
            st.session_state[AGENT_SESSION_KEY] = create_orchestrator_agent(db_path=DB_PATH)
            print("Orchestrator agent object initialized successfully.")
        except Exception as e:
            print(f"Error initializing orchestrator agent: {e}")
            st.error(f"Failed to initialize the AI agent. Please check configurations. Error: {e}")
            st.stop() # Stop execution if agent fails to initialize

# --- Helper Function for History Formatting ---
def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    # Pass the full history including the last user message,
    # as this is a common pattern for chat interfaces.
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history


# --- Agent Interaction ---
def get_agent_response(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Get a response from the agent stored in the session state using the chat method,
    explicitly passing the conversation history.

    Args:
        query: The user's query
        chat_history: The conversation history as a list of ChatMessage objects.

    Returns:
        The agent's response as a string, or an error message.
    """
    if AGENT_SESSION_KEY not in st.session_state or st.session_state[AGENT_SESSION_KEY] is None:
        return "Error: Agent not initialized. Please refresh the page."

    agent = st.session_state[AGENT_SESSION_KEY]

    try:
        # --- Update LLM Temperature from Slider ---
        # Retrieve the temperature value from session state (set by the slider in stui.py)
        # Default to 0.7 if not found, though it should be set by the slider's default
        current_temperature = st.session_state.get("llm_temperature", 0.7)

        # Access the LLM instance within the agent runner's worker and update its temperature
        # AgentRunner stores its worker in `_agent_worker`, and FunctionCallingAgentWorker stores llm in `_llm`
        if hasattr(agent, '_agent_worker') and hasattr(agent._agent_worker, '_llm'):
            actual_llm_instance = agent._agent_worker._llm
            if hasattr(actual_llm_instance, 'temperature'):
                actual_llm_instance.temperature = current_temperature
                print(f"Set LLM temperature to: {current_temperature}")
            else:
                print(f"Warning: LLM object of type {type(actual_llm_instance)} does not have a 'temperature' attribute.")
        else:
            print("Warning: Could not access LLM object within the agent to set temperature. Agent or worker structure might have changed (_agent_worker or _agent_worker._llm not found).")
        # --- End Temperature Update ---

        # Use agent.chat(), passing the explicit history
        with st.spinner("ESI is thinking..."):
            # Pass the formatted history to the agent's chat method
            # The LLM temperature has now been updated for this call
            response = agent.chat(query, chat_history=chat_history)

        # Access the actual response string via the .response attribute
        # The orchestrator agent's response should contain all necessary information,
        # including RAG source markers and code download markers, as per its system prompt.
        response_text = response.response if hasattr(response, 'response') else str(response)

        # The complex manual code execution logic is removed.
        # The Coder Agent is responsible for executing code and providing the ---DOWNLOAD_FILE--- marker.
        # The Orchestrator Agent is responsible for relaying this marker in its final response.
        # stui.py will handle displaying download buttons/images based on these markers in the final response_text.

        # Similarly, RAG source markers (---RAG_SOURCE---) are expected to be included
        # in the orchestrator's final response_text if it used information from the RAG expert.
        # The previous logic for extracting from response.sources can be a fallback,
        # but the primary expectation is that response_text is complete.
        # For now, we rely on the orchestrator's prompt to ensure this.

        print(f"Orchestrator final response text for UI: \n{response_text[:500]}...") # Log snippet
        return response_text

    except Exception as e:
        # Log the error and return a friendly message
        print(f"Error getting orchestrator agent response: {e}")
        print(f"Error getting agent response: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"

def handle_user_input(chat_input_value: str | None):
    """
    Process user input (either from chat box or suggested prompt)
    and update chat with AI response.
    """
    prompt_to_process = None

    # Prioritize suggested prompt if available
    if hasattr(st.session_state, 'prompt_to_use') and st.session_state.prompt_to_use:
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None  # Clear it after using
    # Otherwise, use the value from the chat input box if it's not None
    elif chat_input_value:
        prompt_to_process = chat_input_value

    if prompt_to_process:
        # Add user message to chat history *before* calling the agent
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Format the *entire* current history (including the new user message)
        # to pass to the agent.chat method.
        formatted_history = format_chat_history(st.session_state.messages)

        # Get AI response using the session agent, passing the history
        # This function now includes the manual code execution logic
        response_text = get_agent_response(prompt_to_process, chat_history=formatted_history)


        # Add assistant response (potentially including plot marker from manual execution) to chat history
        # stui.display_chat will handle rendering the text and the plot/download button
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Re-render the chat display which now includes the new message
        # Note: Streamlit reruns the script, so display_chat in create_interface will handle this.
        # We don't need to explicitly call display here.

        # Update suggested prompts based on new chat history (using original full history)
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)

        # --- Persistent storage logic removed ---

        # Force Streamlit to rerun the script immediately to display the new messages
        st.rerun()

# --- UI Callbacks ---
def set_selected_prompt_from_dropdown():
    """Callback function to set the selected prompt from the dropdown."""
    prompt = st.session_state.get("selected_prompt_dropdown", "")
    if prompt:
        st.session_state.prompt_to_use = prompt
        # Optionally clear the dropdown selection after use, though selectbox usually resets visually
        # st.session_state.selected_prompt_dropdown = "" # Might cause issues if not handled carefully


def main():
    """Main function to run the Streamlit app."""
    # --- Initialize LLM Settings FIRST ---
    # This ensures Settings.llm is available for the greeting generation
    try:
        print("Initializing LLM settings...")
        initialize_agent_settings()
        print("LLM settings initialized.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize LLM settings. {e}")
        st.stop()
    # --- End LLM Settings Initialization ---


    # Initialize agent (stores it in session state)
    initialize_agent() # This function now assumes settings are already initialized

    # Handle regeneration request if flag is set
    # This needs to be called before stui.create_interface() so that
    # the messages are updated before being displayed.
    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request() # This function will set do_regenerate to False and rerun

    # Initialize chat history and suggested prompts if they don't exist
    # This is now handled by stui.init_session_state() called within stui.create_interface()

    # Create the rest of the interface using stui (displays chat history, sidebar, etc.)
    # stui.create_interface() will call stui.init_session_state() which handles
    # loading persistent chat history or initializing a new one.
    stui.create_interface() # This displays history and sidebar info

    # Display suggested prompts as a dropdown below the chat history
    # Ensure suggested_prompts is available in session_state (initialized by stui.init_session_state)
    st.selectbox(
        "Select a suggested prompt:",
        # Add a blank option at the beginning
        options=[""] + st.session_state.suggested_prompts,
        key="selected_prompt_dropdown", # Unique key for the selectbox state
        on_change=set_selected_prompt_from_dropdown # Set the callback function
    )

    # Render the chat input box at the bottom, capture its value
    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")

    # Handle user input (either from chat box or a clicked suggested prompt button)
    # If a button was clicked, prompt_to_use is set, otherwise use chat_input_value
    handle_user_input(chat_input_value)


# --- Regeneration Logic ---
def handle_regeneration_request():
    """Handles the request to regenerate the last assistant response."""
    if not st.session_state.get("do_regenerate", False):
        return

    st.session_state.do_regenerate = False # Consume the flag

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'assistant':
        print("Warning: Regeneration called but last message is not from assistant or no messages exist.")
        st.rerun() # Rerun to clear state if needed
        return

    # Case 1: Regenerating the initial greeting
    if len(st.session_state.messages) == 1:
        print("Regenerating initial greeting...")
        # generate_llm_greeting is already imported from agent module
        new_greeting = generate_llm_greeting()
        st.session_state.messages[0]['content'] = new_greeting
        st.rerun()
        return

    # Case 2: Regenerating a response to a user query
    st.session_state.messages.pop() # Remove last assistant message

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'user':
        print("Warning: Cannot regenerate, no preceding user query found after popping assistant message.")
        # Potentially restore the popped message or handle error state more gracefully
        st.rerun()
        return

    print("Regenerating last assistant response to user query...")
    prompt_to_regenerate = st.session_state.messages[-1]['content']
    # The history should include the user message we are responding to
    formatted_history_for_regen = format_chat_history(st.session_state.messages)

    response_text = get_agent_response(prompt_to_regenerate, chat_history=formatted_history_for_regen)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()


if __name__ == "__main__":
    # Display a warning if environment variables are missing
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")

    if not os.path.exists(DB_PATH):
        st.warning(f"⚠️ Database path {DB_PATH} not found. The RAG knowledge base may not be available.")

    main()
