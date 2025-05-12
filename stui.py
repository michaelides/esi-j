import streamlit as st
# import json # Already imported below
import datetime
import os
import re # Import regex module
import json # Added for parsing RAG source JSON
from typing import List, Dict, Any, Optional
# import uuid # No longer needed for persistent session IDs
# Import the new greeting function and DEFAULT_PROMPTS
from agent import generate_llm_greeting, DEFAULT_PROMPTS
# Removed import shutil as it's no longer needed

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_greeting_message() -> str:
    """Generate a greeting message, trying the LLM first and falling back to static."""
    # Attempt to get a dynamic greeting from the LLM
    # This function now resides in agent.py and uses Settings.llm
    # It includes its own error handling and fallback.
    return generate_llm_greeting()

# PERSISTENT_SESSIONS_DIR = "./persistent_sessions" # Removed

def init_session_state():
    """Initialize session state variables for a new chat."""
    # if "persistent_session_id" not in st.session_state: # Removed
    #     st.session_state.persistent_session_id = uuid.uuid4().hex # Removed
    #     print(f"New persistent session ID generated: {st.session_state.persistent_session_id}") # Removed

    # session_id = st.session_state.persistent_session_id # Removed
    # session_file_path = os.path.join(PERSISTENT_SESSIONS_DIR, f"{session_id}.json") # Removed

    if "messages" not in st.session_state: # Initialize messages only if not already set
        # Always start with a fresh greeting message
        print("Initializing new chat session with greeting.")
        st.session_state.messages = [{"role": "assistant", "content": get_greeting_message()}]
        # else: # Removed complex loading logic
            # print(f"No chat history found for session {session_id}. Initializing new chat.") # Removed
            # st.session_state.messages = [{"role": "assistant", "content": get_greeting_message()}] # Removed

    if "suggested_prompts" not in st.session_state:
        # DEFAULT_PROMPTS is now imported at the top of stui.py
        st.session_state.suggested_prompts = DEFAULT_PROMPTS

def display_chat():

    """Display the chat messages from the session state, handling file downloads and image display."""
    CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"  # For files from code interpreter
    RAG_SOURCE_MARKER = "---RAG_SOURCE---"      # For files/links from RAG
    
    # Workspace for code interpreter files, defined relative to PROJECT_ROOT
    CODE_WORKSPACE_RELATIVE = "./code_interpreter_ws"
    code_workspace_absolute = os.path.join(PROJECT_ROOT, CODE_WORKSPACE_RELATIVE)
    os.makedirs(code_workspace_absolute, exist_ok=True)

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # --- Initialize variables for extracted data ---
            text_to_display = content
            rag_sources_data = []
            code_download_filename = None
            code_download_filepath_relative = None # Store relative path from marker
            code_is_image = False

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources ---
                # Use a loop with find to handle multiple RAG sources robustly
                current_pos = 0
                temp_text_for_rag_extraction = text_to_display
                processed_text_after_rag = "" # Build up text without RAG markers
                
                while True:
                    marker_pos = temp_text_for_rag_extraction.find(RAG_SOURCE_MARKER, current_pos)
                    if marker_pos == -1:
                        processed_text_after_rag += temp_text_for_rag_extraction[current_pos:]
                        break 
                        
                    processed_text_after_rag += temp_text_for_rag_extraction[current_pos:marker_pos]
                    
                    # Try to find the end of the JSON payload
                    # Assuming JSON is on a single line after the marker, or clearly delimited
                    json_start_pos = marker_pos + len(RAG_SOURCE_MARKER)
                    # Attempt to parse JSON, allowing for an optional empty line after the marker
                    
                    # First attempt: JSON on the line immediately after the marker (or on the same line if no newline after marker)
                    json_start_pos1 = marker_pos + len(RAG_SOURCE_MARKER)
                    json_end_pos1 = temp_text_for_rag_extraction.find("\n", json_start_pos1)
                    if json_end_pos1 == -1: # Marker is at the end of the text or JSON is last part
                        json_end_pos1 = len(temp_text_for_rag_extraction)
                    
                    json_str = temp_text_for_rag_extraction[json_start_pos1:json_end_pos1].strip()
                    consumed_upto = json_end_pos1 # Default: assume we consumed up to this point

                    # If the first attempt didn't yield a JSON object (e.g., it was an empty line)
                    if not json_str.startswith("{"):
                        # Try the next line
                        json_start_pos2 = json_end_pos1 + 1 # Start looking after the newline of the (potentially empty) first line
                        if json_start_pos2 < len(temp_text_for_rag_extraction): # Check if there is a next line
                            json_end_pos2 = temp_text_for_rag_extraction.find("\n", json_start_pos2)
                            if json_end_pos2 == -1: # JSON is on the last line of the text
                                json_end_pos2 = len(temp_text_for_rag_extraction)
                            
                            potential_json_str_2 = temp_text_for_rag_extraction[json_start_pos2:json_end_pos2].strip()
                            if potential_json_str_2.startswith("{"):
                                json_str = potential_json_str_2 # Use this JSON string
                                consumed_upto = json_end_pos2 # We consumed up to the end of this second line
                            # else: json_str remains from the first attempt (e.g., empty or non-JSON)
                        # else: no second line to check
                    
                    try:
                        if not json_str: # If, after all attempts, json_str is empty
                            raise json.JSONDecodeError("Extracted JSON string is empty after attempts", json_str, 0)
                        
                        rag_data = json.loads(json_str)
                        rag_sources_data.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                        # If JSON is malformed, do NOT add the marker or the failed JSON back to the main text.
                        # The marker and its content are consumed regardless of parsing success.
                        # processed_text_after_rag += temp_text_for_rag_extraction[marker_pos:consumed_upto] # REMOVED THIS LINE

                    current_pos = consumed_upto # Update current_pos to after the consumed part (marker + JSON line(s))

                text_to_display = processed_text_after_rag.strip() # This now contains only text *before* any RAG markers

                # --- 2. Extract Code Interpreter download marker ---
                # This marker is usually at the very end of a specific type of message.
                # Regex search is appropriate here.
                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    # Remove the marker line from the text_to_display
                    text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()
                    
                    print(f"Found code download marker. Filename: {extracted_filename}")
                    # The filename from the marker is relative to the code workspace
                    code_download_filename = extracted_filename
                    code_download_filepath_relative = os.path.join(CODE_WORKSPACE_RELATIVE, extracted_filename) # Store the relative path

                    # Resolve the relative path to an absolute path on the current system
                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        print(f"Code download file exists at: {code_download_filepath_absolute}")
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                            print(f"Detected image file from code interpreter: {code_download_filename}")
                    else:
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_filepath_absolute}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

            # --- 3. Display main text content ---
            if text_to_display:
                st.markdown(text_to_display)

            # --- 4. Display RAG sources (PDFs and Web Links) - Deduplicated ---
            displayed_rag_identifiers = set() # To track displayed sources for this message
            for rag_idx, rag_data in enumerate(rag_sources_data):
                source_type = rag_data.get("type")
                # snippet = rag_data.get("snippet", "") # Snippet display is commented out

                identifier = None
                display_item = False

                if source_type == "pdf":
                    pdf_name = rag_data.get("name", "source.pdf")
                    # The path stored in metadata is now relative
                    pdf_relative_path = rag_data.get("path")
                    
                    identifier = pdf_relative_path # Use relative path for deduplication
                    if identifier and identifier not in displayed_rag_identifiers:
                        # Resolve the relative path to an absolute path on the current system
                        pdf_absolute_path = os.path.join(PROJECT_ROOT, pdf_relative_path) if pdf_relative_path else None

                        if pdf_absolute_path and os.path.exists(pdf_absolute_path):
                            try:
                                citation_num = rag_data.get('citation_number')
                                citation_prefix = f"[{citation_num}] " if citation_num else ""
                                button_label = f"{citation_prefix}Download PDF: {pdf_name}"

                                with open(pdf_absolute_path, "rb") as fp:
                                    st.download_button(
                                        label=button_label,
                                        data=fp,
                                        file_name=pdf_name,
                                        mime="application/pdf",
                                        key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}" # Key still unique per original source item
                                    )
                                print(f"Added download button for RAG PDF: {button_label} (Path: {pdf_absolute_path})")
                                display_item = True
                            except Exception as e:
                                st.error(f"Error creating download button for {pdf_name}: {e}")
                                print(f"Error for RAG PDF '{pdf_name}': {e}")
                        elif pdf_relative_path:
                            # Display filename in warning, not the potentially misleading absolute path
                            st.warning(f"Referenced PDF '{pdf_name}' not found.")
                            print(f"Warning: Referenced PDF '{pdf_name}' not found at expected absolute path: {pdf_absolute_path}")
                
                elif source_type == "web":
                    url = rag_data.get("url")
                    title = rag_data.get("title", url)
                    identifier = url
                    if identifier and identifier not in displayed_rag_identifiers:
                        if url:
                            st.markdown(f"Source: [{title}]({url})")
                            print(f"Added link for RAG web source: {title} (URL: {url})")
                            display_item = True
                
                if display_item and identifier:
                    displayed_rag_identifiers.add(identifier)
                    st.divider() # Show divider only if an item was displayed

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            # Resolve the code download path relative to PROJECT_ROOT
            code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative) if code_download_filepath_relative else None

            if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath): # Ensure file exists before trying to display as image
                try:
                    st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    print(f"Successfully displayed image from code interpreter: {code_download_filename}")
                except Exception as e:
                    st.error(f"Error displaying image {code_download_filename}: {e}")
            
            # If it's an image, it's already displayed. Button is for non-image or if image display fails.
            # Or always show download for images too? For now, only if not displayed as image.
            if code_download_absolute_filepath and (not code_is_image or not os.path.exists(code_download_absolute_filepath)): # Add check if image display failed
                try:
                    with open(code_download_absolute_filepath, "rb") as fp:
                        st.download_button(
                            label=f"Download {code_download_filename}",
                            data=fp,
                            file_name=code_download_filename,
                            mime="application/octet-stream",
                            key=f"code_dl_{msg_idx}_{code_download_filename}"
                        )
                    print(f"Successfully added download button for code interpreter file: {code_download_filename}")
                except Exception as e:
                    st.error(f"Error creating download button for {code_download_filename}: {e}")

            # Add regenerate button for the last assistant message
            if message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1:
                can_regenerate = False
                # Case 1: Initial greeting (assistant is the only message)
                if len(st.session_state.messages) == 1:
                    can_regenerate = True
                # Case 2: Assistant message follows a user message
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True
                
                if can_regenerate:
                    if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                        st.session_state.do_regenerate = True
                        st.rerun()


def create_interface():

    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")


    # Initialize session state
    init_session_state()

    # Create sidebar
    with st.sidebar:
        st.header("About ESI")
        st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
        st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")

        # Suggested prompts moved to main chat area in app.py

        st.divider()

        # Add Temperature Slider
        st.header("LLM Settings")
        # Use a key to store the value in session state
        # Default to a reasonable value like 0.7
        st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("llm_temperature", 0.7), # Get current or default
            step=0.1,
            key="llm_temperature", # Store value in session_state
            help="Controls the randomness of the AI's responses. Lower values are more focused, higher values are more creative."
        )


        st.divider()
        st.info("Made for NBS7091A and NBS7095x")
        
        st.divider()
        if st.button("🔄 Reset Chat", key="reset_chat_button", help="Clears the current conversation and starts a new one."):
            reset_chat_callback()

    # Display chat messages
    display_chat()

def reset_chat_callback():
    """Resets the chat history and suggested prompts to their initial state."""
    print("Resetting chat...")
    # Clear current messages and add the initial greeting
    st.session_state.messages = [{"role": "assistant", "content": get_greeting_message()}]
    # Reset suggested prompts
    st.session_state.suggested_prompts = DEFAULT_PROMPTS # DEFAULT_PROMPTS imported at top
    
    # Clear any pending prompt from dropdown
    if 'prompt_to_use' in st.session_state:
        st.session_state.prompt_to_use = None
    if 'selected_prompt_dropdown' in st.session_state:
        st.session_state.selected_prompt_dropdown = ""

    # Optionally, you might want to clear other session state variables specific to a single chat
    # For now, just messages and prompts are reset. Agent instance and LLM settings persist.
    st.rerun()

# Removed if __name__ == "__main__": block as it's not standard for multipage apps
# and main() is called directly at the end of the script.
# if __name__ == "__main__":
# create_interface()
