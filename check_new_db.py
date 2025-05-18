import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./news_vector_db")
collection = client.get_collection(name="news_articles")

# Retrieve all entries
results = collection.get(include=["metadatas", "documents"])

# Check total entries
total_entries = len(results["ids"])
logger.info(f"Total entries in database: {total_entries}\n")

# List all entries
logger.info("Listing all entries:")
logger.info("=" * 50)

timestamps = []
for id_, metadata, document in zip(results["ids"], results["metadatas"], results["documents"]):
    timestamp = metadata.get("timestamp", "Unknown")
    url = metadata.get("url", "No URL")
    logger.info(f"ID: {id_}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"URL: {url}")
    logger.info(f"Text Preview: {document[:100]}...")
    logger.info("-" * 50)
    timestamps.append(timestamp)

# Check timestamp ordering
if timestamps:
    is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
    if not is_sorted:
        logger.warning("Timestamps are not in sequential order, which may indicate irregular insertion but not necessarily overwrites.")

# Validate entry count (adjust expected range as needed)
if total_entries <= 100:  # Adjust based on your expected max
    logger.info(f"Number of entries ({total_entries}) is within expected range.")
else:
    logger.warning(f"Number of entries ({total_entries}) exceeds expected range.")