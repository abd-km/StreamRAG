import streamlit as st
import torch
import os
import json
import langdetect
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import logging
from langchain.schema import Document

# Workaround for Streamlit-PyTorch issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # Manually set to empty to avoid path resolution error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Hi, Welcome to LiveLens!")

# Set up DeepSeek client using secrets
try:
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
except KeyError:
    st.error("DeepSeek API key not found in secrets. Please set DEEPSEEK_API_KEY in .streamlit/secrets.toml.")
    st.stop()

# Initialize DeepSeek LLM via LangChain
try:
    llm = ChatOpenAI(
        openai_api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        temperature=0.7,
    )
    logger.info("DeepSeek LLM initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize DeepSeek LLM: {str(e)}")
    logger.error(f"DeepSeek LLM error: {str(e)}")
    st.stop()

# Initialize ChromaDB vector store via LangChain
try:
    logger.info("Initializing embeddings...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    logger.info("Initializing ChromaDB vector store...")
    vector_store = Chroma(
        collection_name="trump_news_articles",
        embedding_function=embedding_function,
        persist_directory="/root/SRAG/news_vector_db",
    )
    logger.info("ChromaDB vector store loaded successfully.")
except Exception as e:
    st.error(f"Failed to load ChromaDB vector store: {str(e)}")
    logger.error(f"ChromaDB error: {str(e)}")
    st.stop()

# Define custom prompt template to enforce Trump-only constraint and context usage
qa_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template=(
        "You are an AI assistant. Only respond to questions if they are about Donald Trump. "
        "If the question is not related to Trump, politely reply with "
        "'Sorry, I can only discuss topics related to Donald Trump.'\n"
        "Use the following context as the primary source to inform your response. "
        "Do not rely on your pre-existing knowledge unless the context is insufficient:\n"
        "{context}\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Provide your answer in {{preferred_language}}:\n"
    ),
)

# Set up ConversationalRetrievalChain
try:
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt_template},
    )
    logger.info("ConversationalRetrievalChain initialized successfully.")
    
    # Function to explore vector database content
    def explore_vector_db():
        try:
            # Get all documents from the vector store (increased limit to 195)
            results = vector_store._collection.get(limit=195)
            docs = []
            
            if results and "documents" in results:
                for i, doc_content in enumerate(results["documents"]):
                    metadata = {}
                    if "metadatas" in results and i < len(results["metadatas"]):
                        metadata = results["metadatas"][i]
                    
                    # Try to detect language
                    try:
                        lang = langdetect.detect(doc_content)
                    except:
                        lang = "unknown"
                        
                    docs.append({
                        "content": doc_content[:100] + "..." if len(doc_content) > 100 else doc_content,
                        "metadata": metadata,
                        "language": lang
                    })
            
            return docs
        except Exception as e:
            logger.error(f"Error exploring vector DB: {str(e)}")
            return []
    
    # Function to search for specific topics in the vector database
    def search_vector_db(query, limit=10):
        try:
            docs = vector_store.similarity_search(query, k=limit)
            results = []
            
            for doc in docs:
                # Try to detect language
                try:
                    lang = langdetect.detect(doc.page_content)
                except:
                    lang = "unknown"
                    
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "language": lang
                })
                
            return results
        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            return []
except Exception as e:
    st.error(f"Failed to initialize ConversationalRetrievalChain: {str(e)}")
    logger.error(f"ConversationalRetrievalChain error: {str(e)}")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add sidebar for database exploration
with st.sidebar:
    # Add Al Jazeera SVG logo at the top of sidebar, centered
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("/root/SRAG/Al_Jazeerasvg.svg", width=100, use_container_width=True)
    

    st.header("Vector Database Analysis")
    
    # Add language selection
    st.header("Settings")
    preferred_language = st.selectbox(
        "Response Language",
        ["English", "Spanish", "French", "German", "Arabic", "Chinese", "Urdu", "Bosnian"],
        index=0
    )
    
    # Add direct search functionality
    st.subheader("Direct Vector Search")
    search_query = st.text_input("Search for specific topics:", placeholder="e.g., GCC visit")
    
    if search_query and st.button("Search Database"):
        with st.spinner(f"Searching for '{search_query}'..."):
            results = search_vector_db(search_query, limit=10)
            
            if results:
                st.success(f"Found {len(results)} relevant documents")
                
                for i, doc in enumerate(results):
                    with st.expander(f"Result {i+1} [{doc['language']}]"):
                        st.write(f"**Content**: {doc['content'][:200]}...")
                        st.write(f"**Metadata**: {doc['metadata']}")
            else:
                st.warning("No matching documents found")
    
    st.divider()
    
    # Database overview
    if st.button("Explore All Vector DB Content"):
        with st.spinner("Analyzing vector database content..."):
            docs = explore_vector_db()
            
            st.write(f"Found {len(docs)} documents in the vector store")
            
            # Display language statistics
            languages = {}
            sources = {}
            dates = {}
            
            for doc in docs:
                # Count languages
                lang = doc["language"]
                languages[lang] = languages.get(lang, 0) + 1
                
                # Count sources
                if "source" in doc["metadata"]:
                    source = doc["metadata"]["source"]
                    sources[source] = sources.get(source, 0) + 1
                    
                # Count dates
                if "timestamp" in doc["metadata"]:
                    date = doc["metadata"]["timestamp"].split("T")[0]
                    dates[date] = dates.get(date, 0) + 1
                elif "published" in doc["metadata"]:
                    date = doc["metadata"]["published"].split("T")[0]
                    dates[date] = dates.get(date, 0) + 1
            
            st.subheader("Languages")
            for lang, count in languages.items():
                st.write(f"- {lang}: {count} documents")
                
            st.subheader("Sources")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {source}: {count} documents")
                
            st.subheader("Dates")
            for date, count in sorted(dates.items()):
                st.write(f"- {date}: {count} documents")
            
            # Show sample documents
            st.subheader("Sample Documents")
            for i, doc in enumerate(docs[:5]):
                with st.expander(f"Document {i+1}"):
                    st.write(f"**Content**: {doc['content']}")
                    st.write(f"**Language**: {doc['language']}")
                    st.write(f"**Metadata**: {doc['metadata']}")

# Display messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the ConversationalRetrievalChain
    try:
        logger.info("Querying ConversationalRetrievalChain...")
        # Format prompt template to include preferred language
        formatted_prompt = prompt + f"\n\nPlease respond in {preferred_language}."
        result = qa_chain({
            "question": formatted_prompt,
            "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]
        })
        answer = result["answer"]
        source_docs = result["source_documents"]

        # Log retrieved documents with detailed metadata fallback
        sources = []
        source_details = []
        
        for doc in source_docs:
            metadata = doc.metadata
            content_snippet = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            title = metadata.get("title", f"Content: {content_snippet}")
            source = metadata.get("source", "Unknown Source")
            published = metadata.get("published", metadata.get("timestamp", "Unknown Date"))
            
            # Try to detect language
            try:
                lang = langdetect.detect(doc.page_content)
                lang_name = {
                    'en': 'English',
                    'de': 'German', 
                    'fr': 'French',
                    'es': 'Spanish',
                    'ar': 'Arabic'
                }.get(lang, lang)
            except:
                lang = "unknown"
                lang_name = "Unknown"
            
            # Log all metadata for debugging
            logger.info(f"Document metadata: {metadata}")
            logger.info(f"Document content (first 100 chars): {doc.page_content[:100]}")
            logger.info(f"Detected language: {lang}")
            
            sources.append(f"{title} [{lang_name}] ({source}, {published})")
            source_details.append({
                "title": title,
                "source": source,
                "date": published,
                "language": lang_name,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": metadata
            })
        
        # Display sources to user with more detail
        st.markdown("### Sources:")
        for i, source_detail in enumerate(source_details):
            with st.expander(f"{i+1}. {source_detail['title']} ({source_detail['source']}, {source_detail['date']})"):
                st.markdown(f"**Language**: {source_detail['language']}")
                st.markdown(f"**Content Preview**:")
                st.markdown(f"```\n{source_detail['content']}\n```")
                st.markdown(f"**Metadata**: {json.dumps(source_detail['metadata'], indent=2)}")
            
        logger.info(f"Retrieved sources: {sources}")
        logger.info(f"Answer: {answer[:100]}...")

        # Process answer to add citations
        answer_with_citations = answer
        citation_text = ""
        
        # Add citation markers to the answer if they don't already exist
        for i, source_detail in enumerate(source_details):
            source_marker = f"[{i+1}]"
            citation_text += f"\n\n{source_marker}: {source_detail['title']} ({source_detail['source']}, {source_detail['date']})"
        
        # Only append citations if they don't already exist in the answer
        if "[1]" not in answer_with_citations:
            answer_with_citations = answer + citation_text
        
        # Display answer with citations
        with st.chat_message("assistant"):
            st.markdown(answer_with_citations)

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer_with_citations})

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        logger.error(f"Response generation error: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Sorry, an error occurred while generating the response."
        })