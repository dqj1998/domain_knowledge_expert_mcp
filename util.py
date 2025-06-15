from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from llm_client import generate_answer_with_context
from vectordb import save_to_vector_store

def crawl_site(url: str) -> int:
    """Crawl a website starting from the given URL and return a list of page texts.
    args:
        url (str): The starting URL to crawl.
    returns:
        int: The number of pages crawled.
    example:
        >>> crawl_site("https://example.com")
        ["Page text from https://example.com", "Page text from https://example.com/about"]
    """
    logging.info(f"Starting to crawl site: {url}")
    visited = set()
    to_visit = [url]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue

        res = requests.get(url=current, headers={"Accept": "text/html"})
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text()
        chunks = chunk_texts(text)  # Process one page at a time
        save_to_vector_store(url, chunks) 
        
        visited.add(current)
        max_depth = int(os.getenv("MAX_URL_FOLLOW_DEEPTH", 3))        
        for a in soup.find_all('a', href=True):
            #Compare the / in the current URL and the link URL to determine the depth
            current_depth = current.count('/')
            link_depth = a['href'].count('/')
            if link_depth - current_depth > max_depth:
                continue
            # Join the current URL with the link URL to get the full URL
            full_url = urljoin(current, a['href'])
            if full_url not in visited and should_follow_link(current, full_url):
                to_visit.append(full_url)
    logging.info(f"Finished crawling site: {url}, total pages crawled: {len(visited)}")
    return len(visited)
    
def get_html_text(url: str) -> str:
    """Fetch the HTML content of a URL and return it as text.
    args:
        url (str): The URL to fetch.
    returns:
        str: The text content of the HTML page.
    example:
        >>> get_html_text("https://example.com")
        "HTML text from https://example.com"
    """ 
    res = requests.get(url=url, headers={"Accept": "text/html"})
    if res.status_code != 200:
        logging.error(f"Failed to fetch {url}: {res.status_code}")
        return ""
    if 'text/html' not in res.headers.get('Content-Type', ''):
        logging.error(f"URL {url} does not return HTML content.")
        return ""
    # Parse the HTML content
    if not res.text:
        logging.warning(f"Empty content for URL: {url}")
        return ""
    if not res.text.strip():
        logging.warning(f"Empty content for URL: {url}")
        return ""
    if not res.text.startswith('<'):
        logging.warning(f"Content for URL {url} does not start with '<', indicating it might not be HTML.")
        return res.text
    if not res.text.strip().endswith('>'):
        logging.warning(f"Content for URL {url} does not end with '>', indicating it might not be HTML.")
        return res.text
    soup = BeautifulSoup(res.text, 'html.parser')
    text = soup.get_text()
    return text

def should_follow_link(current_url: str, link_url: str) -> bool:
    """Use LLM to decide whether to follow a link based on the relationship between the current URL's content and the link URL's content."""
    

    # Fetch content from both URLs
    current_content = requests.get(current_url).text
    link_content = requests.get(link_url).text

    # cut the content to avoid too long input
    current_content = current_content[:2000]  # Limit to first 2000 characters
    link_content = link_content[:2000]  # Limit to first 2000 characters
    if not current_content or not link_content:
        logging.warning(f"Empty content for current URL: {current_url} or link URL: {link_url}")
        return False
    
    # Use LLM to judge relationship
    prompt = f"Determine if the content from the link URL is related to the current URL. Should I follow this link?\nCurrent URL Content: {current_content}\nLink URL Content: {link_content}"
    response = generate_answer_with_context(prompt, "")

    # Return True if LLM suggests to follow the link, otherwise False
    return "yes" in response.lower()

def chunk_texts(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def init_logging(log_file_path: str):
    actual_log_file_path = os.getenv("LOG_FILE_PATH", "logs/domain-kg-expert.log.log")

    # Ensure the logs directory exists
    actual_log_dir = os.path.dirname(actual_log_file_path)
    if actual_log_dir: # Ensure dirname is not empty (e.g. if log file is in CWD)
        os.makedirs(actual_log_dir, exist_ok=True)

    # Ensure the log file exists
    if not os.path.exists(actual_log_file_path):
        with open(actual_log_file_path, 'w') as f:
            pass # Just to create it
    
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    # actual_log_file_path is now defined above and will be used by the handler
    log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    log_file_size = int(os.getenv("LOG_FILE_SIZE", 10485760))  # Default to 10MB
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default to 5 backups

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.handlers.RotatingFileHandler(  # Use logging.handlers to access RotatingFileHandler
                actual_log_file_path, maxBytes=log_file_size, backupCount=backup_count
            )
        ]
    )

    # Suppress DEBUG and INFO logs from mcp.server.lowlevel.server in the terminal
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
