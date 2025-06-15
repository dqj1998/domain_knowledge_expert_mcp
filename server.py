from dotenv import load_dotenv
import os
load_dotenv()
if os.getenv("MCP_DEBUG", "false").lower() == "true":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("🧩 Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()

import sys
from mcp.server.fastmcp import FastMCP
from util import init_logging
from util import crawl_site, chunk_texts, should_follow_link, get_html_text
from typing import Dict
import threading
import logging
from sentence_transformers import SentenceTransformer

init_logging("logs/domain-kg-expert.log.log")

mcp = FastMCP("domain-knowledge-export-mcp")

#Try local-all-MiniLM-L6-v2 first, if not exists, download it
try:
    model = SentenceTransformer('local-all-MiniLM-L6-v2')
except Exception as e:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save('./local-all-MiniLM-L6-v2')  # Save manually

from vectordb import init_vector_strore, save_to_vector_store, del_db

@mcp.tool()
def echo(message: str) -> str:
    """Echo tool that returns the input message as is
    Args:
        message (str): The message to echo back.
    Returns:
        str: The echoed message.
    Example:
        >>> echo("Hello, world!")
        "Echo from domain-knowledge-export-mcp: Hello, world!"
    """
    return f"Echo from domain-knowledge-export-mcp: {message}"

@mcp.tool()
def build_kb(url: str) -> str:
    """Build a knowledge base from the given URL.
    Args:
        url (str): The URL to crawl and build the knowledge base from.
    Returns:
        str: A message indicating the progress of the knowledge base building.
    Raises:
        ValueError: If the URL is invalid or if there is an error during crawling.
    Example:
        >>> build_kb("https://example.com")
        "Knowledge base building in progress..."
    """
    if not url.startswith("http"):
        raise ValueError("Invalid URL. Please provide a valid URL starting with 'http'.")

    # if not existing in url_progress_map
    if url in url_progress_map:
        progress = get_progress(url)
        return f"Knowledge base already being built for {url}. Current progress: {progress}%"
    else:
        logging.info(f"Starting to build knowledge base for {url}")
        progress = 0
        update_progress(url, progress)

        # load the URL, genrare summary and save to vector store
        
        text = get_html_text(url)
        if len(text) > 2000:
            text = text[:2000]
        # Generate a summary for the URL by AI
        from llm_client import generate_answer_with_context
        summary = generate_answer_with_context(f"Summarize the content of {url}:", text)
        init_vector_strore(url, summary)

        def start_build_knowledge_thread(url: str):
            thread = threading.Thread(target=build_knowledge_thread, args=(url,))
            thread.start()

        def build_knowledge_thread(url: str):
            progress = build_knowledge(url)
            update_progress(url, progress)

        start_build_knowledge_thread(url)
    return f"Knowledge base building in progress for {url}."

@mcp.tool()
def del_kb(url: str) -> str:
    """Clear the knowledge base for the given URL.
    Args:
        url (str): The URL whose knowledge base should be cleared.
    Returns:
        str: A message indicating that the knowledge base has been cleared.
    Example:
        >>> clear_kb("https://example.com")
        "Knowledge base cleared for https://example.com."
    """
    if url in url_progress_map:
        del url_progress_map[url]
    
    del_db(url)
        
    return f"Knowledge base cleared for {url}."

@mcp.tool()
def ask(prompt: str) -> str:
    """Ask a question to the knowledge base.
    Args:
        prompt (str): The question to ask.
    Returns:
        str: The answer to the question.
    Example:
        >>> ask("What is the capital of France?")
        "The capital of France is Paris."
    """
    from llm_client import generate_answer_with_context
    from vectordb import query_knowledge

    context = query_knowledge(prompt)
    answer = generate_answer_with_context(prompt, context)
    return answer

def build_knowledge(url: str) -> int:
    logging.debug(f"Starting to build knowledge base for {url}")
    pages = crawl_site(url)
    logging.debug(f"Crawled {len(pages)} pages from {url}")

    return 100

# Global map to store the progress for each URL
url_progress_map: Dict[str, int] = {}

def update_progress(url: str, progress: int):
    """Update the progress for a given URL."""
    global url_progress_map
    url_progress_map[url] = progress

def get_progress(url: str) -> int:
    """Get the progress for a given URL"""
    return url_progress_map.get(url, 0)


if __name__ == "__main__":
    """Main entry point"""
    type = sys.argv[1] if len(sys.argv) > 1 else None
    mcp.settings.log_level = os.environ.get("LOG_LEVEL", "DEBUG")
    if type == "sse":
        port = int(os.environ.get("PORT", 3001))
        mcp.settings.port = port
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
    logging.info("Domain Knowledge Export MCP is running.")
    logging.info(f"Listening on port {mcp.settings.port} with host {mcp.settings.host}")