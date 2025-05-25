import streamlit as st
import os
import re
import json
from agent import generate_llm_greeting, DEFAULT_PROMPTS

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_greeting_message() -> str:
    """Generate a greeting message, trying the LLM first and falling back to static."""
    return generate_llm_greeting()

def init_session_state():
    """Initialize session state variables for a new chat."""
    if "messages" not in st.session_state:
        print("Initializing new chat session with greeting.")
        st.session_state.messages = [{"role": "assistant", "content": get_greeting_message()}]

    if "suggested_prompts" not in st.session_state:
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
            code_download_filepath_relative = None
            code_is_image = False

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources ---
                # Regex to find all RAG source markers and capture the JSON string
                # It looks for the marker, then captures everything until a newline or end of string.
                # It also handles potential whitespace/newlines after the marker.
                # Double curly braces {{ and }} are used to include literal { and } in the f-string.
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER)}\s*({{.*?}})\s*(?:\n|$)", re.DOTALL)
                
                # Find all matches
                matches = list(rag_source_pattern.finditer(content))

                # Extract JSON data and remove markers from the content
                text_without_rag_markers = content
                extracted_rag_sources = []
                for match in reversed(matches): # Process in reverse to avoid index issues
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        extracted_rag_sources.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    
                    # Remove the matched marker and its JSON from the text
                    text_without_rag_markers = text_without_rag_markers[:match.start()] + text_without_rag_markers[match.end():]
                
                # Reverse the extracted sources to maintain original order for display
                rag_sources_data = list(reversed(extracted_rag_sources))
                text_to_display = text_without_rag_markers.strip()

                # --- 2. Extract Code Interpreter download marker ---
                # This marker is usually at the very end of a specific type of message.
                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    # Remove the marker line from the text_to_display
                    text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()
                    
                    print(f"Found code download marker. Filename: {extracted_filename}")
                    # The filename from the marker is relative to the code workspace
                    code_download_filename = extracted_filename
                    code_download_filepath_relative = os.path.join(CODE_WORKSPACE_RELATIVE, extracted_filename)

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
            any_rag_sources_displayed = False

            for rag_idx, rag_data in enumerate(rag_sources_data):
                source_type = rag_data.get("type")
                
                identifier = None
                display_item = False

                if source_type == "pdf":
                    pdf_name = rag_data.get("name", "source.pdf")
                    pdf_relative_path = rag_data.get("path")
                    
                    identifier = pdf_relative_path # Use relative path for deduplication
                    if identifier and identifier not in displayed_rag_identifiers:
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
                                        key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                                    )
                                print(f"Added download button for RAG PDF: {button_label} (Path: {pdf_absolute_path})")
                                display_item = True
                            except Exception as e:
                                st.error(f"Error creating download button for {pdf_name}: {e}")
                                print(f"Error for RAG PDF '{pdf_name}': {e}")
                        elif pdf_relative_path:
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
                    any_rag_sources_displayed = True
            
            if any_rag_sources_displayed:
                st.divider()

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative) if code_download_filepath_relative else None

            if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath):
                try:
                    st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    print(f"Successfully displayed image from code interpreter: {code_download_filename}")
                except Exception as e:
                    st.error(f"Error displaying image {code_download_filename}: {e}")
            
            if code_download_absolute_filepath and (not code_is_image or not os.path.exists(code_download_absolute_filepath)):
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

        st.divider()

        # Add Temperature Slider
        st.header("LLM Settings")
        st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("llm_temperature", 0.7),
            step=0.1,
            key="llm_temperature",
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
    st.session_state.suggested_prompts = DEFAULT_PROMPTS
    
    # Clear any pending prompt from dropdown
    if 'prompt_to_use' in st.session_state:
        st.session_state.prompt_to_use = None
    if 'selected_prompt_dropdown' in st.session_state:
        st.session_state.selected_prompt_dropdown = ""

    st.rerun()
