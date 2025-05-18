# StreamRAG: Real-time News Streaming and Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

StreamRAG is a real-time news processing system that combines streaming data ingestion with vector database storage for Retrieval-Augmented Generation (RAG). The system fetches news articles from various sources, processes them, and makes them searchable through both vector similarity search and natural language queries.

## 🏗️ Project Architecture

```
┌─────────────┐    ┌────────┐    ┌──────────┐    ┌───────────────┐
│ Data Source │───▶│ Kafka  │───▶│ Embedder │───▶│ Vector Store  │
│  (News API) │    │        │    │          │    │  (ChromaDB)   │
└─────────────┘    └────────┘    └──────────┘    └───────┬───────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ Frontend UI       │
                                               │ (Streamlit)       │
                                               └───────────────────┘
```

## 🛠️ Components & Technologies

### Core Components

| Filename | Purpose | Technologies | 
|----------|---------|--------------|
| `ingestor.py` | Real-time news article ingestion and processing | Kafka, GoogleNews, SentenceTransformer, ChromaDB |
| `JtoDB.py` | Batch processing from JSON files to vector DB | ChromaDB, SentenceTransformer |
| `populate1.py` | Historical article collection focused on time periods | GoogleNews, Newspaper3k |
| `populate2.py` | Advanced article collection with GDELT API | GDELT API, Threading, Newspaper3k |
| `frontui.py` | Main UI for querying and analyzing news data | Streamlit, LangChain, OpenAI/DeepSeek |
| `check_new_db.py` | Utility to inspect vector database contents | ChromaDB |

### Web Interface

| Filename | Purpose | Technologies |
|----------|---------|--------------|
| `Hackathon-UI/app.py` | Alternative minimal UI for demos | Streamlit, DeepSeek API |

## 🔄 Data Flow & Component Relationships

1. **Data Acquisition**:
   - `ingestor.py`: Streams real-time news via Kafka from Google News API
   - `populate1.py`/`populate2.py`: Batch collect historical news articles

2. **Data Processing & Storage**:
   - All data sources extract article content and metadata
   - Generate embeddings using SentenceTransformer
   - Store vectors and documents in ChromaDB

3. **Data Retrieval & Analysis**:
   - `frontui.py`: Provides UI to query and explore the vector database
   - Performs semantic search and RAG using LLM (DeepSeek)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Kafka server running locally (for real-time mode)
- Required Python packages: `pip install -r requirements.txt`

### Setup and Run

1. **Start Kafka** (for real-time processing):
   ```bash
   ./kafka_2.13-3.9.0/bin/zookeeper-server-start.sh ./kafka_2.13-3.9.0/config/zookeeper.properties
   ./kafka_2.13-3.9.0/bin/kafka-server-start.sh ./kafka_2.13-3.9.0/config/server.properties
   ```

2. **Run Data Processing Pipeline**:
   - For real-time processing: `python ingestor.py`
   - For batch processing from JSON: `python JtoDB.py`
   - To collect news articles: `python populate2.py`

3. **Launch User Interface**:
   ```bash
   python -m streamlit run frontui.py
   ```

## 📊 Key Features

- Real-time news ingestion through Kafka
- Vector embedding for semantic search
- Flexible data sources (Google News, GDELT)
- Streamlit UI for intuitive interaction
- Retrieval-augmented generation with LLMs

## 🔐 Configuration

To use this project, you'll need to set up the following:

1. **API Keys**: Create a `.streamlit/secrets.toml` file containing your LLM API keys:
   ```toml
   DEEPSEEK_API_KEY="your_deepseek_api_key"
   ```

2. **Paths**: Update file paths in the scripts to match your environment.

## 📝 License

All rights reserved.

## 🙏 Acknowledgments
Abdullah Mostafa
Ahmed Al-Ghoul
Ahmed Soliman
Obada Alhomsi
Mohammed Alhato
