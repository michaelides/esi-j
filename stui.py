import streamlit as st
import os
import re
import json
# Removed: from agent import generate_llm_greeting # No longer needed here

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# Removed: get_greeting_message() function as it's now called directly from app.py


def display_chat(DOWNLOAD_MARKER: str, RAG_SOURCE_MARKER_PREFIX: str):
    """Display the chat messages from the session state, handling file downloads and image display."""
    
    # Workspace for code interpreter files, defined relative to PROJECT_ROOT
    CODE_WORKSPACE_RELATIVE = "./code_interpreter_ws"
    code_workspace_absolute = os.path.join(PROJECT_ROOT, CODE_WORKSPACE_RELATIVE)
    os.makedirs(code_workspace_absolute, exist_ok=True)

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # Initialize text_to_display with the full content of the message
            # For assistant messages, content is now expected to be a direct string.
            # For user messages, it's also a string.
            text_to_display = str(content) 
            
            rag_sources_data = []
            code_download_filename = None
            code_download_filepath_relative = None 
            code_is_image = False
            # reasoning_steps is removed as it's no longer part of structured response

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources ---
                # text_to_display is already initialized with the assistant's string content

                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER_PREFIX)}\s*({{.*?}})\s*(?:\n|$)", re.DOTALL)
                matches = list(rag_source_pattern.finditer(text_to_display))
                extracted_rag_sources = []
                for match in reversed(matches):
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        extracted_rag_sources.append(rag_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    text_to_display = text_to_display[:match.start()] + text_to_display[match.end():]
                rag_sources_data = list(reversed(extracted_rag_sources))
                
                # --- 2. Extract Code Interpreter download marker ---
                code_marker_match = re.search(rf"^{re.escape(DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    text_to_display = text_to_display[:code_marker_match.start()] + text_to_display[code_marker_match.end():]
                    code_download_filename = extracted_filename
                    code_download_filepath_relative = os.path.join(CODE_WORKSPACE_RELATIVE, extracted_filename)
                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                    else:
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_filepath_absolute}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"
            
            elif message["role"] == "user":
                pass # No special processing needed here for user role beyond what's done initially
            # Apply strip to the final text to display, after all extractions
            text_to_display = text_to_display.strip()

            # --- 3. Display main text content ---
            # For user messages, this will display their direct input.
            # For assistant messages, this displays the text after marker removal.
            if text_to_display: 
                st.markdown(text_to_display)

            # --- 4. Display RAG sources (PDFs and Web Links) - Deduplicated ---
            # This section only applies if role was assistant and rag_sources_data was populated
            if message["role"] == "assistant" and rag_sources_data: # rag_sources_data is populated only for assistant
                displayed_rag_identifiers = set()
                any_rag_sources_displayed = False
                for rag_idx, rag_data in enumerate(rag_sources_data):
                    source_type = rag_data.get("type")
                    identifier = None
                    display_item = False

                    if source_type == "pdf":
                        pdf_name = rag_data.get("name", "source.pdf")
                        pdf_relative_path = rag_data.get("path")
                        identifier = pdf_relative_path
                        if identifier and identifier not in displayed_rag_identifiers:
                            pdf_absolute_path = os.path.join(PROJECT_ROOT, pdf_relative_path) if pdf_relative_path else None
                            if pdf_absolute_path and os.path.exists(pdf_absolute_path):
                                try:
                                    citation_num = rag_data.get('citation_number')
                                    citation_prefix = f"[{citation_num}] " if citation_num else ""
                                    button_label = f"{citation_prefix}Download PDF: {pdf_name}"
                                    with open(pdf_absolute_path, "rb") as fp:
                                        st.download_button(
                                            label=button_label, data=fp, file_name=pdf_name,
                                            mime="application/pdf", key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                                        )
                                    display_item = True
                                except Exception as e:
                                    st.error(f"Error creating download button for {pdf_name}: {e}")
                            elif pdf_relative_path:
                                st.warning(f"Referenced PDF '{pdf_name}' not found.")
                    
                    elif source_type == "web":
                        url = rag_data.get("url")
                        title = rag_data.get("title", url)
                        identifier = url
                        if identifier and identifier not in displayed_rag_identifiers:
                            if url:
                                st.markdown(f"Source: [{title}]({url})")
                                display_item = True
                    
                    if display_item and identifier:
                        displayed_rag_identifiers.add(identifier)
                        any_rag_sources_displayed = True
                
                if any_rag_sources_displayed:
                    st.divider()

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            # This section only applies if role was assistant and code_download_filepath_relative was set
            if message["role"] == "assistant" and code_download_filepath_relative:
                code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative)
                if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath):
                    try:
                        st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image {code_download_filename}: {e}")
                
                # Changed condition: If it's a download path, and (it's not an image OR the image file doesn't exist)
                if code_download_filename and code_download_absolute_filepath and (not code_is_image or not os.path.exists(code_download_absolute_filepath)):
                    if os.path.exists(code_download_absolute_filepath): # Ensure file exists before trying to open
                        try:
                            with open(code_download_absolute_filepath, "rb") as fp:
                                st.download_button(
                                    label=f"Download {code_download_filename}", data=fp, file_name=code_download_filename,
                                    mime="application/octet-stream", key=f"code_dl_{msg_idx}_{code_download_filename}"
                                )
                        except Exception as e:
                            st.error(f"Error creating download button for {code_download_filename}: {e}")
                    # else: The warning for missing file is handled during marker extraction.

            if message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1:
                can_regenerate = False
                if len(st.session_state.messages) == 1: # If it's the first message (initial greeting)
                    can_regenerate = True
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True
                
                if can_regenerate:
                    if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                        st.session_state.do_regenerate = True 
                        st.rerun()


def _get_chat_as_markdown() -> str:
    """Converts the current chat history to a Markdown string."""
    markdown_content = f"# Discussion: {st.session_state.current_discussion_title}\n\n"
    for message in st.session_state.messages:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"## {role}\n{content}\n\n"
    return markdown_content

def create_interface(DOWNLOAD_MARKER: str, RAG_SOURCE_MARKER_PREFIX: str):
    """Create the Streamlit UI for the chat interface."""

    # --- Sidebar UI ---
    with st.sidebar:
        with st.expander("**Discussion List**", expanded=False, icon = ":material/forum:"): 
            st.info("Conversations are automarically saved and linked to your browser via cookies. Clearing browser data will remove your saved discussions.")
            if not st.session_state.discussion_list:
                st.info("No discussions yet. Start a new one!")
            else:
                for discussion in st.session_state.discussion_list:
                    is_current = (discussion["id"] == st.session_state.current_discussion_id)
                    
                    # Use columns for layout: title/input and options popover
                    col_title, col_options = st.columns([0.8, 0.2])

                    with col_title:
                        if st.session_state.editing_list_discussion_id == discussion['id']:
                            # Display text input for editing
                            st.text_input(
                                "Edit Title",
                                value=discussion['title'],
                                key=f"edit_title_input_{discussion['id']}",
                                on_change=lambda disc_id=discussion['id']: st.session_state._update_listed_discussion_title(disc_id),
                                label_visibility="collapsed" # Hide default label
                            )
                            # No "Done" button here, as the popover will handle exiting edit mode
                            # or clicking another discussion will exit it.
                            # The on_change saves automatically.
                        else:
                            # Display title as a button (for loading or initiating edit via popover)
                            button_label = f"**✦ {discussion['title']}**" if is_current else discussion['title']
                            if st.button(button_label, key=f"select_discussion_{discussion['id']}", use_container_width=True):
                                # If not current, click loads the discussion
                                if not is_current:
                                    st.session_state._load_discussion_session(discussion['id'])
                                # If current, clicking the title button does nothing directly,
                                # editing is handled via the popover.
                                # Removed st.rerun() here. Rely on natural rerun after button click.

                    with col_options:
                        # Popover for options - Changed icon to vertical ellipsis
                        # Removed 'key' argument as st.popover does not accept it directly
                        with st.popover("⋮", use_container_width=True):
                            st.write(f"Options for: **{discussion['title']}**")
                            
                            # Option to edit title
                            if st.button("✏️ Edit Title", key=f"edit_from_popover_{discussion['id']}", use_container_width=True):
                                st.session_state.editing_list_discussion_id = discussion['id']
                                # Removed st.rerun() here. Rely on natural rerun after button click.
                            
                            # Option to download
                            st.download_button(
                                label="⬇️ Download (.md)",
                                data=st.session_state._get_discussion_markdown(discussion['id']), # New function call
                                file_name=f"{discussion['title'].replace(' ', '_')}.md",
                                mime="text/markdown",
                                key=f"download_listed_{discussion['id']}",
                                use_container_width=True
                            )
                            
                            # Option to delete
                            if st.button("♻ Delete", key=f"delete_from_popover_{discussion['id']}", use_container_width=True):
                                st.session_state._delete_current_discussion(discussion['id'])
                                # Removed st.rerun() here. Rely on natural rerun after button click.
            # New Discussion button
            if st.button("➕ New Discussion", use_container_width=True, key="new_discussion_button"):
                st.session_state._create_new_discussion_session()
                # Removed st.rerun() here. Rely on natural rerun after button click.
            # st.divider()

        
        with st.expander("**LLM Settings**", expanded=False, icon = ":material/tune:"):
            st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("llm_temperature", 0.7),
                step=0.1,
                key="llm_temperature",
                help="Controls the randomness of the AI's responses. Lower values are more focused, higher values are more creative."
            )

        # st.divider()
        # st.header("About ESI")

        with st.expander("**About ESI**", expanded=False, icon = ":material/info:"):
          st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
          st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")
          st.info("Made for NBS7091A and NBS7095x")

        CSS = """
        .stExpander > details {
            border: none;
        }
        """
        st.html(f"<style>{CSS}</style>")
    # --- Main Chat Interface ---
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Display chat messages
    display_chat(DOWNLOAD_MARKER, RAG_SOURCE_MARKER_PREFIX)

    # Display suggested prompts as buttons below the chat history
    if st.session_state.suggested_prompts:
        st.markdown("---")
        st.subheader("Suggested Prompts:")
        cols = st.columns(len(st.session_state.suggested_prompts))
        for i, prompt in enumerate(st.session_state.suggested_prompts):
            if cols[i].button(prompt, key=f"suggested_prompt_{i}"):
                st.session_state.prompt_to_use = prompt
                st.rerun() # Trigger rerun to process the prompt
