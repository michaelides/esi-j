import os
import random
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding # Added
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
# from llama_index.core.agent.types import AgentChatResponse # Removed problematic import
from typing import Any # Added for generic type hinting
from llama_index.core.tools import FunctionTool
# from llama_index.core.agent.workflow import ReActAgent # Keep if ReAct is an option later
from google.genai import types # Ensure this import is present
from tools import (
    get_search_tools, 
    get_semantic_scholar_tool_for_agent,
    get_web_scraper_tool_for_agent,
    get_rag_tool_for_agent,
    get_coder_tools
)
from dotenv import load_dotenv
from llama_index.core.llms import LLM # Import base LLM type for type hinting

load_dotenv()


# --- Constants ---
SUGGESTED_PROMPT_COUNT = 4
#llm = Gemini(model_name="models/gemini-2.5-flash-preview-04-17", api_key=os.getenv("GOOGLE_API_KEY"))

# --- Global Settings ---
def initialize_settings():
    """Initializes LlamaIndex settings with Gemini LLM and Embedding model."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Use Google Generative AI Embeddings
    # Default model is "models/embedding-001", ensure GOOGLE_API_KEY is set
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=google_api_key)
    # Use a potentially more stable model name and set a default temperature
    # The temperature can be overridden later based on the slider
    Settings.llm = Gemini(model_name="models/gemini-2.0-flash",
                          api_key=google_api_key,
                          temperature=0.7) # Set default temperature
    # Settings.llm = Gemini(model_name= "models/gemini-2.5-flash-preview-04-17" ,
    #     api_key=google_api_key,
    #     temperature=0.7, # Set default temperature here too if using this model
    #     config=types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=1024)
    #     ),
    # )


# --- Greeting Generation ---
def generate_llm_greeting() -> str:
    """Generates a dynamic greeting message using the configured LLM."""
    static_fallback = "Hello! I'm ESI, your AI assistant for dissertation support. How can I help you today?"
    try:
        # Ensure settings are initialized (might be redundant if called elsewhere first, but safe)
        if not Settings.llm:
            initialize_settings()

        llm = Settings.llm
        if not isinstance(llm, LLM): # Basic check
             print("Warning: LLM not configured correctly for greeting generation.")
             return static_fallback

        # Use a simple prompt for a greeting
        # Using 'complete' for a single, non-chat generation
        prompt = "Generate a short, friendly, and welcoming greeting message (1-2 sentences) for a user interacting with an AI assistant named ESI designed to help with university dissertations. Mention ESI by name."
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


# --- Specialized Agent Definitions ---

def create_search_agent():
    """Creates an agent specialized in web searching."""
    search_tools = get_search_tools()
    if not search_tools:
        print("Warning: No search tools initialized for Search Agent. It will be ineffective.")
        # Return a dummy agent or raise error
        dummy_worker = FunctionCallingAgentWorker.from_tools([], llm=Settings.llm, verbose=True, system_prompt="I am a search agent but have no tools.")
        return AgentRunner(dummy_worker)


    system_prompt = """You are a specialized search agent.
Your sole purpose is to use the provided search tools (DuckDuckGo, Tavily, Wikipedia) to find information on the web in response to a query.
Provide comprehensive information and cite sources/URLs clearly in your response.
If you cannot find relevant information, state that clearly.
Do not attempt to answer questions outside of this scope or use tools not provided."""
    
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=search_tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )
    return AgentRunner(agent_worker)

def create_literature_reviewer_agent():
    """Creates an agent specialized in finding academic literature."""
    lit_tools = get_semantic_scholar_tool_for_agent()
    if not lit_tools:
        print("Warning: No literature review tools initialized. Literature Reviewer Agent will be ineffective.")
        dummy_worker = FunctionCallingAgentWorker.from_tools([], llm=Settings.llm, verbose=True, system_prompt="I am a literature reviewer but have no tools.")
        return AgentRunner(dummy_worker)

    system_prompt = """You are a specialized literature reviewer agent.
Your sole purpose is to use the Semantic Scholar tool to find academic papers relevant to the given query.
Provide details of the papers found, including titles, authors, and summaries if available.
If no relevant papers are found, state that clearly.
Do not attempt to answer questions outside of this scope or use tools not provided."""

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=lit_tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )
    return AgentRunner(agent_worker)

def create_scraper_agent():
    """Creates an agent specialized in scraping web content from URLs."""
    scraper_tools = get_web_scraper_tool_for_agent()
    if not scraper_tools:
        print("Warning: No scraper tool initialized. Scraper Agent will be ineffective.")
        dummy_worker = FunctionCallingAgentWorker.from_tools([], llm=Settings.llm, verbose=True, system_prompt="I am a web scraper but have no tool.")
        return AgentRunner(dummy_worker)

    system_prompt = """You are a specialized web scraper agent.
Your sole purpose is to use the web_scraper tool to fetch the main textual content from a given URL.
The input to you will be the URL to scrape.
Return the scraped text. If you encounter an error (e.g., URL not accessible, no content found), report the error clearly.
Do not attempt to answer questions outside of this scope or use tools not provided."""
    
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=scraper_tools, # Should be a list with one tool
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )
    return AgentRunner(agent_worker)

def create_rag_agent(db_path):
    """Creates an agent specialized in RAG from a local knowledge base."""
    # get_rag_tool_for_agent returns a single tool (FunctionTool)
    rag_tool_function = get_rag_tool_for_agent(db_path=db_path, collection_name="resources")
    if not rag_tool_function: # This function itself is the tool, not a list
        print("Warning: RAG tool failed to initialize. RAG Agent will be ineffective.")
        dummy_worker = FunctionCallingAgentWorker.from_tools([], llm=Settings.llm, verbose=True, system_prompt="I am a RAG agent but have no RAG tool.")
        return AgentRunner(dummy_worker)

    system_prompt = """You are a specialized RAG (Retrieval Augmented Generation) agent.
Your sole purpose is to answer questions based *only* on the information available in the local dissertation knowledge base, accessed via the `rag_dissertation_retriever` tool.
If the information is not found in the knowledge base, state that clearly. Do not attempt to guess or use external knowledge.
When you retrieve and use information, your response MUST include the exact source markers (e.g., `---RAG_SOURCE---{...json...}`) provided by the tool. Ensure these markers are on their own lines in your output.
Format your answer clearly based on the retrieved documents."""
    
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[rag_tool_function], # The RAG tool is a FunctionTool, needs to be in a list
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )
    return AgentRunner(agent_worker)

def create_coder_agent():
    """Creates an agent specialized in writing and executing Python code."""
    coder_tools = get_coder_tools()
    if not coder_tools:
        print("Warning: No coder tools initialized. Coder Agent will be ineffective.")
        dummy_worker = FunctionCallingAgentWorker.from_tools([], llm=Settings.llm, verbose=True, system_prompt="I am a coder agent but have no tools.")
        return AgentRunner(dummy_worker)

    system_prompt = """You are a specialized coder agent.
Your purpose is to write and execute Python code to solve problems, perform calculations, or generate data/files as requested.
You have access to a `code_interpreter` tool.
- When your Python code needs to save a file (e.g., a plot, a CSV file), it MUST save the file directly using its filename (e.g., `plt.savefig('plot.png')`, `df.to_csv('data.csv')`). Do NOT create subdirectories or include paths in the save command. The file will be saved in the tool's pre-configured workspace (`./code_interpreter_ws`).
- After your code execution confirms a file has been saved, your final response MUST include a special marker on its own line: `---DOWNLOAD_FILE---filename.ext`, replacing `filename.ext` with the actual name of the saved file (e.g., `---DOWNLOAD_FILE---plot.png`).
- You should also include any standard output (stdout) and standard error (stderr) from the code execution in your response for the user to see.
- Present the Python code you executed clearly, formatted using Markdown (e.g., ```python\n# your code here\n```).
If a task does not require code execution or file generation, state that.
If there's an error during code execution, report it clearly.
"""
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=coder_tools, # This is already a list of tools from CodeInterpreterToolSpec
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )
    return AgentRunner(agent_worker)

# --- Orchestrator Agent ---
def create_orchestrator_agent(db_path="./ragdb/chromadb"):
    """Creates the Orchestrator Agent that delegates to specialized agents."""
    initialize_settings() # Ensure LLM settings are initialized

    # Initialize specialized agents
    search_agent = create_search_agent()
    lit_reviewer_agent = create_literature_reviewer_agent()
    scraper_agent = create_scraper_agent()
    rag_agent_instance = create_rag_agent(db_path=db_path) # Renamed to avoid conflict
    coder_agent_instance = create_coder_agent() # Renamed to avoid conflict

    # Convert specialized agents into tools for the orchestrator
    # Create wrapper functions for each agent's chat method to ensure simple string input schema
    expert_tools = []

    if search_agent:
        def search_agent_chat_wrapper(message: str) -> Any:
            return search_agent.chat(message)
        expert_tools.append(FunctionTool.from_defaults(
            fn=search_agent_chat_wrapper,
            name="search_expert",
            description="Delegates queries to a specialized web search agent. Use for finding general information, news, current events, or broad topics on the internet. Input must be a single string query."
        ))

    if lit_reviewer_agent:
        def lit_reviewer_agent_chat_wrapper(message: str) -> Any:
            return lit_reviewer_agent.chat(message)
        expert_tools.append(FunctionTool.from_defaults(
            fn=lit_reviewer_agent_chat_wrapper,
            name="literature_expert",
            description="Delegates queries to a specialized literature reviewer agent. Use for finding academic papers and scholarly articles via Semantic Scholar. Input must be a single string query."
        ))

    if scraper_agent:
        def scraper_agent_chat_wrapper(message: str) -> Any:
            return scraper_agent.chat(message)
        expert_tools.append(FunctionTool.from_defaults(
            fn=scraper_agent_chat_wrapper,
            name="scraper_expert",
            description="Delegates tasks to a specialized web scraper agent. Use to fetch the textual content of a specific webpage URL. Input must be a single URL string."
        ))

    if rag_agent_instance:
        def rag_agent_chat_wrapper(message: str) -> Any:
            return rag_agent_instance.chat(message)
        expert_tools.append(FunctionTool.from_defaults(
            fn=rag_agent_chat_wrapper,
            name="rag_expert",
            description="Delegates queries to a specialized RAG agent. Use for queries about specific institutional knowledge, previously saved research, or topics likely covered in the local dissertation knowledge base. Use this first for university-specific questions. Input must be a single string query."
        ))

    if coder_agent_instance:
        def coder_agent_chat_wrapper(message: str) -> Any:
            return coder_agent_instance.chat(message)
        expert_tools.append(FunctionTool.from_defaults(
            fn=coder_agent_chat_wrapper,
            name="coder_expert",
            description="Delegates tasks to a specialized coder agent. Use to write and execute Python code for tasks like data analysis, plotting, complex calculations, or file generation. Input must be a single string describing the task."
        ))

    if not expert_tools:
        raise RuntimeError("No expert agent tools could be initialized for the Orchestrator. Orchestrator cannot function.")

    try:
        with open("esi_agent_instruction.md", "r") as f:
             system_prompt_base = f.read().strip()
    except FileNotFoundError:
        print("Warning: esi_agent_instruction.md not found. Using default base prompt for orchestrator.")
        system_prompt_base = "You are ESI, an AI assistant for dissertation support."

    orchestrator_system_prompt = f"""{system_prompt_base}
Your role is to understand the user's query and delegate tasks to a team of specialized expert agents to gather information, then synthesize a comprehensive final answer for the user.
You have access to the following expert agents as tools:
*   `search_expert`: For general web searches, current events, or broad topics. Input should be the search query string.
*   `literature_expert`: For finding academic papers and scholarly articles. Input should be the research query string.
*   `scraper_expert`: To fetch the content of a specific webpage URL. Input should be the URL string.
*   `rag_expert`: For queries about specific institutional knowledge, previously saved research, or topics likely covered in the local dissertation knowledge base. Use this first for university-specific questions. Input should be the query string for the knowledge base.
*   `coder_expert`: To write and execute Python code for tasks like data analysis, plotting, or complex calculations. Input should be a description of the task for the coder.

Your process:
1.  Analyze the user's query carefully.
2.  Determine which expert agent(s) are best suited to handle the query or parts of it. You can call multiple experts sequentially if needed.
3.  Formulate clear and concise sub-questions or tasks as input for each chosen expert agent tool.
4.  Call the expert agent(s) using the provided tools.
5.  Review the responses from the expert agent(s).
6.  Synthesize all gathered information into a single, coherent, and helpful final answer for the user.
7.  **Crucially, you MUST ensure the following markers from expert agents are included in YOUR final synthesized response to the user, if the information is used**:
    *   If the `rag_expert` provides `---RAG_SOURCE---{{...}}` markers in its response, you MUST include these exact markers (each on its own new line) in your final answer.
    *   If the `coder_expert`'s response indicates a file has been saved and provides a `---DOWNLOAD_FILE---filename.ext` marker, you MUST include this exact marker (on its own new line) at the end of your final answer.
    *   If `coder_expert` provides Python code it executed, include this code in your final response, formatted with Markdown (e.g., ```python\ncode here\n```).

Be proactive, thorough, and cite sources when possible (based on information from experts).
If an expert agent returns an error or no useful information, acknowledge this politely. You may try another expert or ask the user for clarification.
Your final output to the user should be a single, complete response. Do not expose your internal thought process or delegation strategy to the user in the final response unless it's helpful for context (e.g., "I searched the web and found...").
"""

    orchestrator_worker = FunctionCallingAgentWorker.from_tools(
        tools=expert_tools,
        llm=Settings.llm,
        system_prompt=orchestrator_system_prompt,
        verbose=True, # Set to False for production
    )
    orchestrator_agent_runner = AgentRunner(orchestrator_worker)
    print(f"Orchestrator agent created with {len(expert_tools)} expert tools.")
    return orchestrator_agent_runner

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
        # Ensure LLM is initialized
        if not Settings.llm:
            print("Warning: LLM not initialized for suggested prompts. Returning defaults.")
            return DEFAULT_PROMPTS

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


# if __name__ == '__main__':
#     # Example usage (for testing)
#     from dotenv import load_dotenv
#     load_dotenv()
#     try:
#         # Test Orchestrator Agent
#         # Ensure DB_PATH is correctly pointing to your chromadb for RAG agent
#         test_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragdb', 'chromadb'))
#         print(f"Using test DB path for RAG agent: {test_db_path}")
        
#         orchestrator = create_orchestrator_agent(db_path=test_db_path)
#         print("Orchestrator agent created successfully.")

#         # Example query for the orchestrator
#         # response = orchestrator.chat("Find recent papers on AI in education and summarize one.")
#         # print("\nOrchestrator Response:", response)

#         # response_rag = orchestrator.chat("What is the university policy on plagiarism? (uses RAG)")
#         # print("\nOrchestrator RAG Response:", response_rag)
        
#         # response_code = orchestrator.chat("Generate a simple plot of y=x^2 and save it as plot.png")
#         # print("\nOrchestrator Coder Response:", response_code)
#         # Check ./code_interpreter_ws/plot.png after this

#         # Test prompt generation (still uses global Settings.llm)
#         sample_history = [
#             {"role": "user", "content": "Tell me about research ethics."},
#             {"role": "assistant", "content": "Research ethics involves principles like informed consent..."}
#         ]
#         prompts = generate_suggested_prompts(sample_history)
#         print("\nSuggested Prompts:", prompts)


#     except Exception as e:
#         print(f"Error during agent testing: {e}")
#         import traceback
#         traceback.print_exc()
