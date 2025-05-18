import json
import os
import datetime
import time
import re
import sys
import argparse
import requests
import hashlib
import random
from typing import Dict, List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from newspaper import Article, Config
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Settings
OUTPUT_FILE = "gdelt_trump_articles.json"
MAX_WORKERS = 3  # Limits parallel requests
TIMEOUT = 30
MAX_PARAGRAPHS = 2
SMART_PARAGRAPHS = True
MAX_RESULTS = 50  # Limit results per query

# Browser config
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config.request_timeout = 15
config.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml',
    'Accept-Language': 'en-US,en;q=0.9',
}
config.ignore_ssl_errors = True

# Common stopwords
ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 
    'for', 'with', 'about', 'to', 'in', 'on', 'this', 'that', 'it', 'is', 'was', 'be', 'are', 
    'of', 'as', 'has', 'have', 'had', 'not', 'what', 'all', 'will'
}

def format_date_readable(iso_date: str) -> str:
    """Format ISO date string into human-readable format"""
    if not iso_date:
        return "Unknown Date"
    
    try:
        # Convert to datetime and format
        dt = datetime.datetime.fromisoformat(iso_date)
        return dt.strftime("%B %d, %Y at %H:%M")
    except Exception as e:
        logger.warning(f"Error formatting date {iso_date}: {e}")
        return iso_date  # Use original on error

def get_domain_from_url(url: str) -> str:
    """Get website domain from URL"""
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        return parsed_url.netloc
    except:
        return url

def get_text_fingerprint(text: str) -> str:
    """Generate a fingerprint for text content to detect duplicates"""
    # Normalize text: lowercase, remove punctuation, extra spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hash it
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def select_best_paragraphs(paragraphs: List[str], max_paragraphs: int) -> str:
    """Get the most relevant paragraphs"""
    if not paragraphs or len(paragraphs) <= max_paragraphs:
        return '\n\n'.join(paragraphs)
    
    # Take first paragraphs if not many
    if len(paragraphs) <= max_paragraphs + 2:
        return '\n\n'.join(paragraphs[:max_paragraphs])
    
    try:
        # First paragraphs usually contain key info
        return '\n\n'.join(paragraphs[:max_paragraphs])
    
    except Exception as e:
        logger.warning(f"Error selecting paragraphs: {e}. Using first {max_paragraphs} instead.")
        return '\n\n'.join(paragraphs[:max_paragraphs])

def extract_article_from_url(url: str, language: str = 'en', max_paragraphs: int = 2, smart_paragraphs: bool = True) -> Optional[Dict[str, Any]]:
    """Download and extract article text"""
    try:
        # Create article object
        article = Article(url, language=language)
        
        # Get article content
        article.download()
        article.parse()
        
        # Check if article has content
        if not article.title or not article.text:
            logger.warning(f"Article at {url} has no title or text. Skipping.")
            return None
        
        # Check if article is about Trump
        if "trump" not in article.title.lower() and "trump" not in article.text.lower():
            logger.info(f"Article at {url} doesn't mention Trump. Skipping.")
            return None
        
        # Extract paragraphs
        paragraphs = [p.strip() for p in article.text.split('\n\n') if p.strip()]
        
        # Use smart paragraph selection if enabled
        if smart_paragraphs:
            limited_content = select_best_paragraphs(paragraphs, max_paragraphs)
        else:
            limited_content = '\n\n'.join(paragraphs[:max_paragraphs])
        
        # Extract key fields
        title = article.title
        source = article.source_url or get_domain_from_url(url)
        published_iso = article.publish_date.isoformat() if article.publish_date else datetime.datetime.now().isoformat()
        
        # Format date in a readable way
        readable_date = format_date_readable(published_iso)
        
        # Create embedding text that combines all information in a formatted way
        embedding_text = f"Title: {title}\n\nPublished: {readable_date}\n\nSource: {source}\n\nContent: {limited_content}"
        
        # Create article dictionary matching trump_articles.json format with embedding_text
        article_data = {
            "title": title,
            "content": limited_content,
            "source": source,
            "url": url,
            "published": published_iso,
            "embedding_text": embedding_text
        }
        
        return article_data
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

def query_gdelt_api(query: str, max_results: int = 100, max_retries: int = 5) -> List[str]:
    """Search for articles via GDELT API"""
    from urllib.parse import quote
    
    # Encode query for URL
    encoded_query = quote(query)
    
    # GDELT API endpoint
    gdelt_api = f"https://api.gdeltproject.org/api/v2/doc/doc?query={encoded_query}&mode=artlist&format=json&maxrecords={max_results}"
    
    # Retry with exponential backoff
    for retry in range(max_retries):
        try:
            logger.info(f"Querying GDELT API with: {query} (attempt {retry+1}/{max_retries})")
            logger.info(f"Full URL: {gdelt_api}")
            
            # Add a randomized user agent to avoid detection
            headers = {
                'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/{100+retry}.0.{4000+retry}.{100+retry}',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(gdelt_api, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract URLs
            urls = []
            if "articles" in data:
                for article in data["articles"]:
                    if "url" in article:
                        urls.append(article["url"])
            
            logger.info(f"Found {len(urls)} URLs in GDELT API response")
            return urls
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                if retry < max_retries - 1:
                    # Exponential backoff: 5s, 10s, 20s, 40s...
                    wait_time = 5 * (2 ** retry)
                    logger.warning(f"Rate limited (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} retries: {e}")
            else:
                logger.error(f"HTTP Error: {e}")
                break
                
        except Exception as e:
            logger.error(f"Error querying GDELT API: {e}")
            if retry < max_retries - 1:
                time.sleep(5)  # Simple delay for other errors
            else:
                break
    
    return []

def process_url(url: str, existing_fingerprints: Set[str], language: str = 'en') -> Optional[Dict[str, Any]]:
    """Process URL and check for duplicates"""
    try:
        # Get article content
        article_data = extract_article_from_url(url, language, MAX_PARAGRAPHS, SMART_PARAGRAPHS)
        
        # Skip if extraction failed
        if article_data is None:
            return None
        
        # Create fingerprint to check for duplicates
        content_for_fingerprint = article_data["title"] + " " + article_data["content"]
        fingerprint = get_text_fingerprint(content_for_fingerprint)
        
        # Skip duplicates
        if fingerprint in existing_fingerprints:
            logger.info(f"Skipping duplicate article: {article_data['title']}")
            return None
        
        # Save fingerprint and return article
        existing_fingerprints.add(fingerprint)
        return article_data
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None

def load_existing_articles(filepath: str) -> List[Dict[str, Any]]:
    """Load articles from JSON"""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {filepath}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles: {e}")
        return []

def get_existing_fingerprints(articles: List[Dict[str, Any]]) -> Set[str]:
    """Generate fingerprints from articles"""
    fingerprints = set()
    
    for article in articles:
        if "title" in article and "content" in article:
            content = article["title"] + " " + article["content"]
            fingerprint = get_text_fingerprint(content)
            fingerprints.add(fingerprint)
    
    return fingerprints

def save_articles(articles: List[Dict[str, Any]], filepath: str) -> None:
    """Save articles to JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(articles)} articles to {filepath}")

def main():
    """Main script - collect news articles"""
    parser = argparse.ArgumentParser(description="Collect Trump news from GDELT API")
    parser.add_argument("--queries", nargs="+", default=["trump"], help="Search queries (default: 'trump')")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS, help=f"Results per query (default: {MAX_RESULTS})")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--merge", action="store_true", help="Include trump_articles.json data")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load existing data if needed
    existing_articles = []
    if args.append and os.path.exists(args.output):
        existing_articles = load_existing_articles(args.output)
    
    # Import from trump_articles.json if merging
    if args.merge and os.path.exists("trump_articles.json"):
        trump_articles = load_existing_articles("trump_articles.json")
        existing_articles.extend(trump_articles)
        logger.info(f"Added {len(trump_articles)} articles from trump_articles.json")
    
    # Create fingerprints for deduplication
    existing_fingerprints = get_existing_fingerprints(existing_articles)
    logger.info(f"Got {len(existing_fingerprints)} article fingerprints")
    
    # Run all search queries
    all_urls = []
    for query in args.queries:
        # Wait between queries to avoid rate limits
        if all_urls:  # Skip delay on first query
            delay = random.uniform(10, 15)
            logger.info(f"Waiting {delay:.1f} seconds before next query...")
            time.sleep(delay)
        
        urls = query_gdelt_api(query, args.max_results)
        all_urls.extend(urls)
        logger.info(f"Found {len(urls)} URLs for query: {query}")
    
    # Remove duplicate URLs
    unique_urls = list(set(all_urls))
    logger.info(f"Found {len(unique_urls)} unique URLs from {len(all_urls)} total")
    
    # Process URLs in parallel
    processed_articles = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit jobs
        future_to_url = {
            executor.submit(process_url, url, existing_fingerprints): url 
            for url in unique_urls
        }
        
        # Process results as they complete
        for future in future_to_url:
            url = future_to_url[future]
            try:
                article = future.result(timeout=TIMEOUT)
                if article is not None:
                    processed_articles.append(article)
                    
                    # Print progress
                    sys.stdout.write(f"\rCollected {len(processed_articles)} articles...")
                    sys.stdout.flush()
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
    
    print()  # New line after progress indicator
    
    # Combine with existing articles if in append mode
    if args.append or args.merge:
        all_articles = existing_articles + processed_articles
    else:
        all_articles = processed_articles
    
    # Save all articles
    save_articles(all_articles, args.output)
    
    # Show results
    execution_time = time.time() - start_time
    logger.info(f"Finished in {execution_time:.2f} seconds")
    logger.info(f"Collected {len(processed_articles)} new articles")
    logger.info(f"Total articles saved: {len(all_articles)}")

if __name__ == "__main__":
    main()
