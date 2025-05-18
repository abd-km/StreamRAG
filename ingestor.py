import time
import json
import chromadb
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading
import logging
import uuid
from GoogleNews import GoogleNews
import pandas as pd
from dateutil import parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./news_vector_db")
collection = client.get_or_create_collection(name="news_articles")

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'news_articles'

# Initialize Kafka producer
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5
    )
except KafkaError as e:
    logger.error(f"Failed to initialize Kafka producer: {e}")
    exit(1)

# Google News configuration
REQUEST_INTERVAL = 120  # 30 seconds
MAX_REQUESTS_PER_DAY = 100  # Conservative limit for testing
REQUEST_COUNT = 0
ARTICLES_PER_FETCH = 3  # Limit to 5 articles per fetch

# Fetch news from Google News
def fetch_news():
    global REQUEST_COUNT
    if REQUEST_COUNT >= MAX_REQUESTS_PER_DAY:
        logger.warning("Reached daily request limit. Stopping fetches.")
        return []

    try:
        # User input for topic (only on the first fetch)
        if REQUEST_COUNT == 0:
            user_request = input("🔍 Enter a topic to search in Google News: ")
        else:
            # Reuse the same topic for subsequent fetches
            user_request = getattr(fetch_news, 'topic', None)
            if not user_request:
                user_request = input("🔍 Enter a topic to search in Google News: ")
        fetch_news.topic = user_request

        # Initialize GoogleNews for the past 7 days
        googlenews = GoogleNews(period='1d')
        googlenews.search(user_request)

        all_results = []
        max_pages = 100  # Try up to 100 pages

        # Loop to get multiple pages until we have enough articles
        for i in range(1, max_pages + 1):
            googlenews.getpage(i)
            results = googlenews.result()
            if not results:
                break  # Stop if no more results
            all_results.extend(results)
            logger.info(f"📄 Page {i}: {len(results)} results, total: {len(all_results)}")

            # Stop once we have enough articles to select 5
            if len(all_results) >= ARTICLES_PER_FETCH:
                break

            time.sleep(0.5)  # Be polite to avoid rate limiting

        # Convert to DataFrame and remove duplicates within this fetch
        df = pd.DataFrame(all_results)
        df.drop_duplicates(subset='title', inplace=True)

        # Parse date to datetime
        df['datetime'] = df['date'].apply(lambda x: parser.parse(x, fuzzy=True) if pd.notnull(x) else pd.NaT)

        # Final structured data
        df = df[['datetime', 'title', 'media', 'link', 'desc']].sort_values(by='datetime', ascending=False).reset_index(drop=True)

        # Limit to 5 articles per fetch
        df = df.head(ARTICLES_PER_FETCH)

        REQUEST_COUNT += 1
        logger.info(f"Fetch {REQUEST_COUNT}/{MAX_REQUESTS_PER_DAY}")
        logger.info(f"Fetched {len(df)} unique articles within this fetch")

        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        return []

# Stream news to Kafka
def news_producer(stop_event):
    # Load existing titles from ChromaDB at startup
    existing_titles = set()
    try:
        results = collection.get(include=["documents"])
        for doc in results["documents"]:
            # Extract title from the document (assuming title is the first line)
            title = doc.split('\n\n')[0].strip()
            existing_titles.add(title)
        logger.info(f"Loaded {len(existing_titles)} existing titles from ChromaDB.")
    except Exception as e:
        logger.warning(f"Failed to load existing titles from ChromaDB: {e}. Starting with empty set.")

    while not stop_event.is_set():
        articles = fetch_news()
        for article in articles:
            # Extract relevant fields
            title = article.get('title', '')
            description = article.get('desc', '')  # 'desc' instead of 'description' from GoogleNews
            url = article.get('link', '')
            published_at = article.get('datetime', datetime.now().isoformat())
            if pd.notna(published_at):
                published_at = published_at.isoformat()
            else:
                published_at = datetime.now().isoformat()
            source = article.get('media', '')

            # Combine title and description for embedding
            text = f"{title}\n\n{description}" if description else title
            if not text.strip():
                continue

            # Check for duplicate title in ChromaDB and current session
            if title in existing_titles:
                logger.info(f"Skipping duplicate article (in ChromaDB): {title}")
                continue
            existing_titles.add(title)

            # Generate unique ID
            article_id = f"news_{uuid.uuid4()}"

            # Create message
            message = {
                'id': article_id,
                'text': text,
                'metadata': {
                    'timestamp': published_at,
                    'source': source,
                    'url': url
                }
            }

            # Send to Kafka
            try:
                producer.send(KAFKA_TOPIC, message)
                logger.info(f"Sent article {article_id} to Kafka")
            except KafkaError as e:
                logger.error(f"Failed to send to Kafka: {e}")

        # Wait for next fetch
        for _ in range(int(REQUEST_INTERVAL)):
            if stop_event.is_set():
                break
            time.sleep(1)

# Consume from Kafka and store in ChromaDB
def news_consumer(stop_event):
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='latest',
        group_id='news_consumer_group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    try:
        for message in consumer:
            if stop_event.is_set():
                break

            data = message.value
            article_id = data['id']
            text = data['text']
            metadata = data['metadata']

            # Generate embedding
            try:
                embedding = embedder.encode(text).tolist()
            except Exception as e:
                logger.error(f"Failed to embed article {article_id}: {e}")
                continue

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

    except KafkaError as e:
        logger.error(f"Kafka consumer error: {e}")
    finally:
        consumer.close()

# Main function
def main():
    # Create stop event
    stop_event = threading.Event()

    # Start producer thread
    producer_thread = threading.Thread(target=news_producer, args=(stop_event,))
    producer_thread.daemon = True
    producer_thread.start()

    # Start consumer thread
    consumer_thread = threading.Thread(target=news_consumer, args=(stop_event,))
    consumer_thread.daemon = True
    consumer_thread.start()

    try:
        # Run for 30 minutes
        time.sleep(1800)
    except KeyboardInterrupt:
        logger.info("Stopping simulation...")
    finally:
        stop_event.set()
        producer_thread.join(timeout=2)
        consumer_thread.join(timeout=2)
        producer.close()
        logger.info("Simulation complete. Data stored in ./news_vector_db")

if __name__ == "__main__":
    main()
