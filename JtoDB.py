import json
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the JSON file
JSON_FILE_PATH = Path("C:/Code/Notebook/articles.json")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./news_vector_db")
collection = client.get_or_create_collection(name="news_articles")

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_articles():
    """Load articles from the JSON file."""
    try:
        if not JSON_FILE_PATH.exists():
            logger.error(f"JSON file not found at {JSON_FILE_PATH}")
            return []
        
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            logger.error("JSON file must contain a list of articles")
            return []
        
        logger.info(f"Loaded {len(articles)} articles from {JSON_FILE_PATH}")
        return articles
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return []

def embed_and_store_articles():
    """Embed articles and store them in ChromaDB."""
    articles = load_articles()
    if not articles:
        logger.info("No articles to process. Exiting.")
        return

    for article in articles:        # Extract fields
        title = article.get('title', '')
        content = article.get('content', '')
        source = article.get('source', '')
        url = article.get('url', '')
        published = article.get('published', '')
        
        # Use embedding_text if available, otherwise combine title and content
        if 'embedding_text' in article and article['embedding_text'].strip():
            text = article['embedding_text']
        else:
            text = f"{title}\n\n{content}" if content else title
        if not text.strip():
            logger.warning("Skipping article with empty text")
            continue

        # Generate unique ID to avoid overwriting
        article_id = f"news_{uuid.uuid4()}"

        # Generate embedding
        try:
            embedding = embedder.encode(text).tolist()
        except Exception as e:
            logger.error(f"Failed to embed article {article_id}: {e}")
            continue        # Prepare metadata
        metadata = {
            'timestamp': published,
            'source': source,
            'url': url,
            'title': title
        }

        # Store in ChromaDB
        try:
            collection.add(
                ids=[article_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            logger.info(f"Stored article {article_id} in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to store article {article_id}: {e}")

def main():
    embed_and_store_articles()
    logger.info("Processing complete. Data stored in ./news_vector_db")

if __name__ == "__main__":
    main()