import os
import asyncio
# import chromadb # Removed chromadb
from urllib.parse import urlparse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.vector_stores.chroma import ChromaVectorStore # Removed chromadb
from llama_index.core.vector_stores import SimpleVectorStore # Added SimpleVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding # Added
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from dotenv import load_dotenv
import sys # Import sys to check Python version if needed, or for sys.exit

# Load environment variables from a .env file if it exists
load_dotenv()

# Determine project root based on the script's location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
# Define the path for the SimpleVectorStore persistence
SIMPLE_STORE_PERSIST_PATH = os.path.join(project_root, "ragdb", "simple_vector_store")
print(f"Simple Vector Store Persistence Path (make_rag): {SIMPLE_STORE_PERSIST_PATH}") # Debug output
# collection_name = "resources" # No longer needed for SimpleVectorStore
# LlamaIndex uses slightly different model names sometimes, adjust if needed
CHUNK_SIZE = 512 # Adjusted chunk size, common for LlamaIndex
CHUNK_OVERLAP = 20 # Adjusted chunk overlap
# Define the directory containing source documents for the RAG database
SOURCE_DATA_DIR = os.path.join(project_root, "ragdb/source_data") # Use absolute path
WEB_MARKDOWN_PATH = os.path.join(project_root, "ragdb/web_markdown") # Use absolute path

# --- Add URLs to scrape ---
WEBPAGES_FILE = os.path.join(project_root, 'ragdb/webpages.txt')
URLS_TO_SCRAPE = []
try:
    with open(WEBPAGES_FILE, 'r') as file:
        # Strip whitespace/newlines from each line
        URLS_TO_SCRAPE = [line.strip() for line in file if line.strip()]
    if not URLS_TO_SCRAPE:
        print(f"Warning: {WEBPAGES_FILE} is empty. No webpages will be scraped.")
except FileNotFoundError:
    print(f"Warning: Could not find {WEBPAGES_FILE}. Please create this file in the project root directory and add URLs to scrape, one per line. No webpages will be scraped.")
    # Continue without scraping if file is missing
except Exception as e:
    print(f"Error reading {WEBPAGES_FILE}: {e}. No webpages will be scraped.")
    # Continue without scraping if error occurs


# --- Embedding Model (Initialized in main based on args) ---

def url_to_filename(url: str, max_length: int = 200) -> str:
    """Converts a URL to a safe filename for storing markdown, truncating if necessary."""
    parsed_url = urlparse(url)
    # Start with netloc and path
    filename_parts = []
    if parsed_url.netloc:
        filename_parts.append(parsed_url.netloc)
    if parsed_url.path:
        # Remove leading/trailing slashes from path before replacing
        path_part = parsed_url.path.strip('/')
        if path_part: # Add path if it's not just "/"
            filename_parts.append(path_part)

    filename_base = "_".join(filename_parts)

    # Replace common problematic characters not suitable for filenames
    # Allow alphanumeric, underscore, hyphen, dot. Replace others with underscore.
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. " # Added space for temp replacement
    filename_base = "".join(c if c in safe_chars else '_' for c in filename_base)
    filename_base = filename_base.replace(' ', '_') # Replace spaces with underscores

    # Remove consecutive underscores
    while "__" in filename_base:
        filename_base = filename_base.replace("__", "_")

    # Truncate if too long (before adding extension)
    # Reserve space for ".md" extension
    if len(filename_base) > max_length - 3:
        filename_base = filename_base[:max_length - 3]

    # Remove any leading/trailing underscores that might have resulted
    filename_base = filename_base.strip('_')

    if not filename_base: # Handle edge case of empty base (e.g. "http://example.com" might become just "example_com")
        # Fallback to a generic name or hash if needed, here using netloc or "default"
        filename_base = parsed_url.netloc.replace('.', '_').replace(':', '_') if parsed_url.netloc else "scraped_page"
        if len(filename_base) > max_length -3: # Truncate fallback too
            filename_base = filename_base[:max_length -3]
        filename_base = filename_base.strip('_')


    return f"{filename_base}.md"


async def scrape_websites(urls, output_dir):
    """Scrapes a list of URLs and saves the content as markdown files."""
    print("Running web scraping...")
    os.makedirs(output_dir, exist_ok=True) # Ensure output_dir itself exists
    print(f"Output directory for scraped markdown: {output_dir}")

    if not urls:
        print("No URLs provided for scraping.")
        return

    print(f"Initializing crawler for deep scraping of {len(urls)} seed URLs...")
    # Initialize the crawler once.
    # should_markdown=True implies LXMLWebScrapingStrategy by default,
    # but we'll be explicit in CrawlerRunConfig.
    crawler = AsyncWebCrawler(
        should_markdown=True
        # Consider adding other AsyncWebCrawler parameters if needed
    )

    print("Starting deep web scraping...")
    saved_count = 0
    failed_urls_details = []
    # Define deep crawl parameters
    MAX_DEPTH = 1  # Start page (depth 0) + 1 level of links
    MAX_PAGES_PER_SEED = 10 # Max pages to crawl per seed URL
    INCLUDE_EXTERNAL = False

    try:
        for start_url in urls:
            print(f"  Initiating deep crawl from: {start_url} (max_depth={MAX_DEPTH}, max_pages_per_seed={MAX_PAGES_PER_SEED})")
            
            config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=MAX_DEPTH,
                    include_external=INCLUDE_EXTERNAL,
                    max_pages=MAX_PAGES_PER_SEED
                ),
                scraping_strategy=LXMLWebScrapingStrategy(), # Be explicit
                stream=True, # Process results as they come
                verbose=True # More output from the crawler
            )

            try:
                async for result in await crawler.arun(url=start_url, config=config):
                    current_page_url = result.url
                    print(f"    Processing page from deep crawl: {current_page_url} (Origin: {start_url}, Depth: {result.metadata.get('depth', 'N/A')})")

                    try:
                        if result.success and result.markdown:
                            filename = url_to_filename(current_page_url)
                            filepath = os.path.join(output_dir, filename)
                            try:
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(result.markdown)
                                print(f"      Successfully scraped and saved: {current_page_url} to {filepath}")
                                saved_count += 1
                            except IOError as e:
                                print(f"      Error writing file {filepath} for URL {current_page_url}: {e}")
                                failed_urls_details.append({"url": current_page_url, "error": f"File write error: {e}"})
                        elif result.success and not result.markdown:
                            print(f"      Successfully processed URL: {current_page_url}, but no markdown content was returned.")
                            # Optionally log this as a "failed" item if content is expected
                            # failed_urls_details.append({"url": current_page_url, "error": "Deep crawl succeeded for page but no markdown content."})
                        else:
                            error_message = result.error or "Unknown error during deep crawl of page"
                            print(f"      Failed to scrape page {current_page_url} during deep crawl: {error_message}")
                            failed_urls_details.append({"url": current_page_url, "error": error_message})
                    except Exception as page_processing_e:
                        print(f"      Error processing result for {current_page_url}: {page_processing_e}")
                        failed_urls_details.append({"url": current_page_url, "error": f"Result processing error: {page_processing_e}"})
            
            except Exception as e:
                # This catches errors from the arun() call itself for the start_url,
                # or errors setting up the crawl for this start_url.
                print(f"    An error occurred while initiating or performing deep crawl for start URL {start_url}: {e}")
                failed_urls_details.append({"url": start_url, "error": f"Deep crawl initiation/execution error: {str(e)}"})
    finally:
        if 'crawler' in locals() and hasattr(crawler, 'close'):
            print("Closing crawler session...")
            await crawler.close()

    # The 'len(urls)' here refers to seed URLs. The actual number of pages attempted might be higher.
    print(f"Deep scraping finished. Successfully saved {saved_count} pages from {len(urls)} seed URLs.")
    if failed_urls_details:
        print("Details for failed URLs:")
        for detail in failed_urls_details:
            print(f"  URL: {detail['url']}, Error: {detail['error']}")

    print("Finished web scraping.")

# --- Main Script ---
async def main():
    # --- Initialize Embedding Model ---
    # Initialize the embedding model here, it will be passed to the index
    embedding_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

    print(f"Target Persistence Path: {SIMPLE_STORE_PERSIST_PATH}")


    # 1. Scrape websites
    await scrape_websites(URLS_TO_SCRAPE, WEB_MARKDOWN_PATH)

    # 2. Load documents using SimpleDirectoryReader
    print(f"Loading documents from {SOURCE_DATA_DIR} and {WEB_MARKDOWN_PATH}...")
    # Ensure directories exist before loading
    os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
    os.makedirs(WEB_MARKDOWN_PATH, exist_ok=True)

    # List of directories to read from
    input_dirs = [SOURCE_DATA_DIR, WEB_MARKDOWN_PATH]
    # Required file extensions (add more if needed, ensure dependencies like pypdf are installed)
    required_exts = [".pdf", ".txt", ".md", ".csv"]

    all_documents = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            print(f"Directory is empty or does not exist, skipping: {input_dir}")
            continue
        try:
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                required_exts=required_exts,
                recursive=True, # Scan subdirectories
                # file_metadata=filename_fn # Optional: Add metadata based on filename
            )
            docs = reader.load_data(show_progress=True)
            if docs:
                print(f"Loaded {len(docs)} documents from {input_dir}")
                all_documents.extend(docs)
            else:
                 print(f"No documents with extensions {required_exts} found in {input_dir}")
        except ValueError as e:
            # SimpleDirectoryReader raises ValueError if input_dir doesn't exist (handled above)
            print(f"Warning: Error reading from directory {input_dir}: {e}")
        except Exception as e:
            print(f"Error loading documents from {input_dir}: {e}")


    print(f"Total documents loaded: {len(all_documents)}")

    if not all_documents:
        print("No documents loaded. Exiting.")
        return

    # 3. Initialize Text Splitter (Node Parser)
    print(f"Initializing node parser (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    # Use LlamaIndex SentenceSplitter
    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 4. Initialize SimpleVectorStore and Storage Context
    print("Initializing SimpleVectorStore...")
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Create VectorStoreIndex (This performs parsing, embedding, and indexing)
    # This approach processes all documents every time.
    print(f"Creating index for {len(all_documents)} documents... (This may take a while)")
    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=embedding_model, # Pass the selected embedding model
        node_parser=node_parser, # Pass the configured node parser
        show_progress=True,
    )

    # 6. Persist the index, vector store, and other data
    print(f"Persisting index to disk at {SIMPLE_STORE_PERSIST_PATH}...")
    os.makedirs(SIMPLE_STORE_PERSIST_PATH, exist_ok=True)
    index.storage_context.persist(persist_dir=SIMPLE_STORE_PERSIST_PATH)

    print(f"Successfully created and persisted index to {SIMPLE_STORE_PERSIST_PATH}")
    # Verification is implicit in successful persistence. We can't easily get a 'count' like with Chroma.

    print("RAG database creation script finished.")


if __name__ == "__main__":
   # Run the async main function
   asyncio.run(main())

