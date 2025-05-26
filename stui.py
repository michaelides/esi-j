import streamlit as st
import os
import re
import json
from agent import generate_llm_greeting, DEFAULT_PROMPTS
# user_data_manager is not directly imported here anymore,
# as its functions are called via st.session_state in app.py

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_greeting_message() -> str:
    """Generate a greeting message, trying the LLM first and falling back to static."""
    return generate_llm_greeting()

# init_session_state is no longer needed here as app.py handles it
# def init_session_state():
#     """Initialize session state variables for a new chat."""
#     pass

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
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER)}\s*({{.*?}})\s*(?:\n|$)", re.DOTALL)
                
                matches = list(rag_source_pattern.finditer(content))

                text_without_rag_markers = content
                extracted_rag_sources = []
                for match in reversed(matches):
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        extracted_rag_sources.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    
                    text_without_rag_markers = text_without_rag_markers[:match.start()] + text_without_rag_markers[match.end():]
                
                rag_sources_data = list(reversed(extracted_rag_sources))
                text_to_display = text_without_rag_markers.strip()

                # --- 2. Extract Code Interpreter download marker ---
                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()
                    
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
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_filepath_absolute}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

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
                        st.session_state.do_regenerate = True
                        st.rerun()

def _on_discussion_selection_change():
    """Callback for the discussion selection dropdown."""
    selected_discussion = st.session_state.selected_discussion_dropdown
    if selected_discussion and selected_discussion["id"] is not None:
        # An existing discussion was selected
        st.session_state._load_discussion_session(selected_discussion["id"])
    elif selected_discussion and selected_discussion["id"] is None:
        # "➕ New Discussion" was selected
        st.session_state._create_new_discussion_session()


def create_interface():
    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Display current discussion title
    if st.session_state.get("current_discussion_id"):
        st.text_input("Discussion Title", value=st.session_state.discussion_title, key="discussion_title_input", on_change=_update_discussion_title)
    
    # Create sidebar
    with st.sidebar:
        # User Account section is removed from here, now handled in app.py main()
        # and only shows a generic message about cookie persistence.

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
        # Discussions section moved here
        st.header("Discussions")
        if st.session_state.user_info: # This check should always be true now
            # Dropdown to select existing discussions
            discussion_options = [{"id": None, "title": "➕ New Discussion"}] + st.session_state.all_discussions
            
            # Find the index of the current discussion in the options list
            current_discussion_index = 0
            if st.session_state.current_discussion_id:
                for i, disc in enumerate(discussion_options):
                    if disc["id"] == st.session_state.current_discussion_id:
                        current_discussion_index = i
                        break

            st.selectbox(
                "Select or create a discussion:",
                options=discussion_options,
                format_func=lambda x: x["title"],
                key="selected_discussion_dropdown",
                index=current_discussion_index,
                on_change=_on_discussion_selection_change # Call the new function in stui.py
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Current", key="save_discussion_button", help="Save the current discussion."):
                    st.session_state._save_current_discussion() # Call app.py's function
                    st.success("Discussion saved!")
            with col2:
                if st.button("🗑️ Delete Current", key="delete_discussion_button", help="Delete the current discussion. This cannot be undone."):
                    if st.session_state.current_discussion_id:
                        st.session_state._delete_current_discussion() # Call app.py's function
                    else:
                        st.warning("No discussion selected to delete.")
            
            st.markdown("---")
            st.subheader("Download Current Discussion")
            st.download_button(
                label="Download as Markdown",
                data=_get_chat_as_markdown(),
                file_name=f"{st.session_state.discussion_title.replace(' ', '_')}.md",
                mime="text/markdown",
                key="download_markdown_button"
            )
            # Placeholder for other download options
            # st.button("Download as PDF (Coming Soon)", disabled=True)
            # st.button("Download as DOCX (Coming Soon)", disabled=True)

        st.divider()
        if st.button("🔄 Reset Current Chat", key="reset_chat_button", help="Clears the current conversation and starts a new one, saving the old one."):
            st.session_state.reset_chat_callback() # Call app.py's function
        st.divider()
        st.info("Made for NBS7091A and NBS7095x")
        
    
    # Display chat messages
    display_chat()

def _update_discussion_title():
    """Callback to update the discussion title when the text input changes."""
    new_title = st.session_state.discussion_title_input
    if new_title != st.session_state.discussion_title:
        st.session_state.discussion_title = new_title
        st.session_state._save_current_discussion() # Save to update title in file
        st.session_state._refresh_discussion_list() # Refresh list to show new title in dropdown
        print(f"Discussion title updated to: {new_title}")

def _get_chat_as_markdown() -> str:
    """Converts the current chat history to a Markdown string."""
    markdown_content = f"# Discussion: {st.session_state.discussion_title}\n\n"
    for message in st.session_state.messages:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"## {role}\n{content}\n\n"
    return markdown_content
