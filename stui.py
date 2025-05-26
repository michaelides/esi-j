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
            text_to_display = content
            rag_sources_data = []
            code_download_filename = None
            code_download_filepath_relative = None
            code_is_image = False

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources ---
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER_PREFIX)}\s*({{.*?}})\s*(?:\n|$)", re.DOTALL)
                
                matches = list(rag_source_pattern.finditer(text_to_display)) # Search in current text_to_display

                extracted_rag_sources = []
                # Iterate in reverse to avoid issues with index shifts when removing parts
                for match in reversed(matches):
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        extracted_rag_sources.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    
                    # Remove the marker and its content from text_to_display
                    text_to_display = text_to_display[:match.start()] + text_to_display[match.end():]
                
                rag_sources_data = list(reversed(extracted_rag_sources))
                
                # --- 2. Extract Code Interpreter download marker ---
                # Search in the potentially modified text_to_display
                code_marker_match = re.search(rf"^{re.escape(DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    # Remove the marker from text_to_display
                    text_to_display = text_to_display[:code_marker_match.start()] + text_to_display[code_marker_match.end():]
                    
                    print(f"Found code download marker. Filename: {extracted_filename}")
                    code_download_filename = extracted_filename
                    code_download_filepath_relative = os.path.join(CODE_WORKSPACE_RELATIVE, extracted_filename)

                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        print(f"Code download file exists at: {code_download_filepath_absolute}")
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                            print(f"Detected image file from code interpreter: {code_download_filename}")
                    else:
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_absolute_filepath}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

            # Apply strip to the final text to display, after all extractions
            text_to_display = text_to_display.strip()

            # --- 3. Display main text content ---
            if text_to_display:
                st.markdown(text_to_display)

            # --- 4. Display RAG sources (PDFs and Web Links) - Deduplicated ---
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
                if len(st.session_state.messages) == 1:
                    can_regenerate = True
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True
                
                if can_regenerate:
                    if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                        st.session_state.do_regenerate = True # Set flag for app.py to handle
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
        st.header("User Account")
        st.write(f"Logged in as: **Guest User**") # User ID is internal, not displayed as name
        st.info("Your conversations are automatically saved and linked to your browser. Clearing browser data may remove your saved discussions.")
        
        st.divider()
        
        st.header("Discussions")
        # New Discussion button
        if st.button("➕ New Discussion", use_container_width=True, key="new_discussion_button"):
            st.session_state._create_new_discussion_session()
            # st.rerun() is called by _create_new_discussion_session

        # Removed the old "Current Discussion Title" editing block
        # This block is now replaced by inline editing in the "Your Discussions" list below.
        # if st.session_state.editing_discussion_title:
        #     st.text_input(...)
        #     if st.button("✅ Done Editing", ...):
        #         st.session_state.editing_discussion_title = False
        #         st.rerun()
        # else:
        #     if st.button(f"### {st.session_state.current_discussion_title}", ...):
        #         st.session_state.editing_discussion_title = True
        #         st.rerun()

        st.subheader("Your Discussions")
        if not st.session_state.discussion_list:
            st.info("No discussions yet. Start a new one!")
        else:
            for discussion in st.session_state.discussion_list:
                is_current = (discussion["id"] == st.session_state.current_discussion_id)
                
                # Use columns for layout: title/input and delete button
                # Adjust column width for better display of title and delete button
                col1, col2 = st.columns([0.8, 0.2])

                with col1:
                    if st.session_state.editing_list_discussion_id == discussion['id']:
                        # Display text input for editing
                        st.text_input(
                            "Edit Title",
                            value=discussion['title'],
                            key=f"edit_title_input_{discussion['id']}",
                            on_change=lambda disc_id=discussion['id']: st.session_state._update_listed_discussion_title(disc_id),
                            label_visibility="collapsed" # Hide default label
                        )
                        # Add a "Done" button to exit edit mode for this specific item
                        if st.button("✅ Done", key=f"done_edit_{discussion['id']}", use_container_width=True):
                            st.session_state.editing_list_discussion_id = None
                            st.rerun()
                    else:
                        # Display title as a button (for loading or initiating edit)
                        # Use markdown to style the button label, especially for the current discussion
                        button_label = f"**▶️ {discussion['title']}**" if is_current else discussion['title']
                        if st.button(button_label, key=f"select_or_edit_{discussion['id']}", use_container_width=True):
                            if is_current:
                                # If current, click toggles edit mode for this item
                                st.session_state.editing_list_discussion_id = discussion['id']
                            else:
                                # If not current, click loads the discussion
                                st.session_state._load_discussion_session(discussion['id'])
                            st.rerun()
                
                with col2:
                    # Delete button for each listed discussion
                    # Ensure the delete button is aligned and visible
                    if st.button("🗑️", key=f"delete_listed_{discussion['id']}", help=f"Delete '{discussion['title']}'"):
                        # This will delete the specific discussion clicked
                        st.session_state._delete_current_discussion(discussion['id']) # Pass the ID to delete
                        # st.rerun() is called by _delete_current_discussion

        st.markdown("---")
        st.subheader("Download Current Discussion")
        st.download_button(
            label="Download as Markdown",
            data=_get_chat_as_markdown(),
            file_name=f"{st.session_state.current_discussion_title.replace(' ', '_')}.md", # Standardized name
            mime="text/markdown",
            key="download_markdown_button"
        )
        # Placeholder for other download options
        # st.button("Download as PDF (Coming Soon)", disabled=True)
        # st.button("Download as DOCX (Coming Soon)", disabled=True)

        st.divider()

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
        st.header("About ESI")
        st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
        st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")
        st.info("Made for NBS7091A and NBS7095x")
        
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
