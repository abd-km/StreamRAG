from GoogleNews import GoogleNews
from newspaper import Article, Config
import pandas as pd
from dateutil import parser
import json
import time
import random
import re
from urllib.parse import unquote
import os
import datetime

# Format dates to be readable (like May 16, 2025)
def format_date_readable(iso_date):
    if not iso_date:
        return "Unknown Date"
    
    try:
        dt = datetime.datetime.fromisoformat(iso_date)
        return dt.strftime("%B %d, %Y at %H:%M")
    except Exception as e:
        print(f"Error formatting date {iso_date}: {e}")
        return iso_date  # Just use original if parsing fails

# Settings
SEARCH_QUERY = "trump"
TARGET_ARTICLES = 400  # Target article count
MAX_BATCH_SIZE = 100   # Max articles per batch
MAX_RETRIES = 2        # Number of download attempts
SAVE_INTERVAL = 50     # Save every 50 articles
OUTPUT_FILE = "historical_trump_articles.json"

# Browser config
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config.request_timeout = 15
config.ignore_ssl_errors = True  # Ignore SSL cert issues

# Search periods (2016-2024)
HISTORICAL_PERIODS = [
    # 2016 - Campaign and election
    {"from": "01/01/2016", "to": "03/31/2016"},
    {"from": "04/01/2016", "to": "06/30/2016"},
    {"from": "07/01/2016", "to": "09/30/2016"},
    {"from": "10/01/2016", "to": "12/31/2016"},
    
    # 2017 - First year as president
    {"from": "01/01/2017", "to": "06/30/2017"},
    {"from": "07/01/2017", "to": "12/31/2017"},
    
    # 2018-2019 - Mid-term presidency
    {"from": "01/01/2018", "to": "06/30/2018"},
    {"from": "07/01/2018", "to": "12/31/2018"},
    {"from": "01/01/2019", "to": "06/30/2019"},
    {"from": "07/01/2019", "to": "12/31/2019"},
    
    # 2020 - Election year
    {"from": "01/01/2020", "to": "03/31/2020"},
    {"from": "04/01/2020", "to": "06/30/2020"},
    {"from": "07/01/2020", "to": "09/30/2020"},
    {"from": "10/01/2020", "to": "12/31/2020"},
    
    # 2021-2023 - Post-presidency
    {"from": "01/01/2021", "to": "06/30/2021"},
    {"from": "07/01/2021", "to": "12/31/2021"},
    {"from": "01/01/2022", "to": "12/31/2022"},
    {"from": "01/01/2023", "to": "12/31/2023"},
    
    # 2024 - Campaign and election  
    {"from": "01/01/2024", "to": "06/30/2024"},
    {"from": "07/01/2024", "to": "12/31/2024"}
]

# Clean URLs by removing tracking params
def clean_url(url):
    if isinstance(url, str):
        url = re.split(r'/&ved|&ved', url)[0]
        url = unquote(url)  # Decode URL
        url = url.replace(' ', '%20').replace('|', '%7C')
    return url

# Keep just the first few paragraphs
def limit_paragraphs(text):
    paragraphs = text.split('\n\n')
    
    if len(paragraphs) <= 2:
        limited_text = text
    else:
        limited_text = paragraphs[0] + '\n\n' + paragraphs[1]
        
        # Add a third paragraph if the second is short
        if len(paragraphs[1]) < 100 and len(paragraphs) > 2:
            limited_text += '\n\n' + paragraphs[2]
    
    return limited_text

# Extract article content from URL
def extract_article(url, config):
    for attempt in range(MAX_RETRIES):
        try:
            url = clean_url(url)
            
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            text = article.text.strip()
            
            if not text or len(text) < 100:
                raise ValueError("Content too short or empty")
            
            limited_text = limit_paragraphs(text)
            
            return limited_text, None
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = random.uniform(2, 5)
                print(f"⚠️ Attempt {attempt+1} failed. Retrying in {delay:.1f}s... Error: {str(e)[:100]}")
                time.sleep(delay)
            else:
                return None, str(e)
    
    return None, "Max retries exceeded"

# Collect news articles from different time periods
def collect_historical_news(query, target_count):
    all_articles = []
    successful_articles = 0
    failed_urls = []
    total_processed = 0
    period_index = 0
    
    # Resume from previous run if file exists
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_articles = json.load(f)
                successful_articles = len(all_articles)
                print(f"Loaded {successful_articles} existing articles from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error loading existing file: {e}")
    
    print(f"Collecting {target_count} articles about '{query}'")
    
    # Shuffle time periods for variety
    random.shuffle(HISTORICAL_PERIODS)
    
    while successful_articles < target_count and period_index < len(HISTORICAL_PERIODS) * 2:
        # Select time period - cycle through available periods
        current_period = HISTORICAL_PERIODS[period_index % len(HISTORICAL_PERIODS)]
        period_index += 1
        
        print(f"\n📊 Progress: {successful_articles}/{target_count} articles collected")
        print(f"🔍 Searching for '{query}' in period: {current_period['from']} to {current_period['to']}")
        
        # Initialize GoogleNews with current period
        googlenews = GoogleNews(lang='en')
        googlenews.set_time_range(current_period['from'], current_period['to'])
        googlenews.search(query)
        
        # Track articles for this batch
        batch_results = []
        page_num = 1
        
        # Get as many pages as needed until we have enough articles or run out of results
        while len(batch_results) < MAX_BATCH_SIZE:
            print(f"📄 Getting page {page_num}...")
            googlenews.getpage(page_num)
            results = googlenews.result()
            
            if not results:
                print("No more results available from this period.")
                break
                
            batch_results.extend(results)
            print(f"   Found {len(results)} articles on page {page_num}, total in batch: {len(batch_results)}")
            
            page_num += 1
            
            # Random delay between page requests
            time.sleep(random.uniform(3, 5))
        
        # Process batch results
        if batch_results:
            df = pd.DataFrame(batch_results)
            total_processed += len(df)
            
            # Remove duplicate titles
            if 'title' in df.columns:
                df.drop_duplicates(subset=['title'], keep='first', inplace=True)
            
            # Skip already processed URLs
            if 'link' in df.columns:
                existing_urls = [article.get('url', '') for article in all_articles]
                df = df[~df['link'].isin(existing_urls)]
            
            # Process each article
            batch_successful = 0
            batch_failed = 0
            
            print(f"\nProcessing {len(df)} articles in this batch...")
            
            for idx, row in df.iterrows():
                url = row.get('link', '')
                
                # Skip if URL is empty
                if not url:
                    batch_failed += 1
                    continue
                    
                print(f"Processing article {successful_articles+1}/{target_count} (batch: {idx+1}/{len(df)}): {url[:60]}...")
                
                # Extract article content
                content, error = extract_article(url, config)
                
                if content:
                    # Article successfully extracted
                    title = row.get('title', '')
                    source = row.get('media', '')
                    published_date = row['datetime'].isoformat() if pd.notnull(row.get('datetime')) else None
                    
                    # Create readable published date format for embedding
                    readable_date = format_date_readable(published_date)
                    
                    # Create embedding text that combines all information in a readable format
                    embedding_text = f"Title: {title}\n\nPublished: {readable_date}\n\nSource: {source}\n\nContent: {content}"
                    
                    article_data = {
                        "title": title,
                        "content": content,
                        "source": source,
                        "url": url,
                        "published": published_date,
                        "embedding_text": embedding_text
                    }
                    
                    all_articles.append(article_data)
                    batch_successful += 1
                    successful_articles += 1
                    print(f"✓ Success ({len(content)} chars)")
                    
                    # Save intermediate progress
                    if successful_articles % SAVE_INTERVAL == 0:
                        save_to_json(all_articles, OUTPUT_FILE)
                        print(f"🔄 Progress saved: {successful_articles}/{target_count} articles")
                        
                else:
                    # Article failed to extract
                    batch_failed += 1
                    failed_urls.append(url)
                    print(f"✗ Failed: {error}")
                
                # Check if we've reached our target
                if successful_articles >= target_count:
                    break
                
                # Random delay between article processing
                time.sleep(random.uniform(1.5, 3.5))
            
            print(f"\nBatch complete: {batch_successful} succeeded, {batch_failed} failed")
        
        else:
            print("No articles found in this period, trying another period...")
        
        # Clear GoogleNews results before next period
        googlenews.clear()
        
        # Wait between period changes
        wait_time = random.uniform(10, 20)
        print(f"Waiting {wait_time:.1f} seconds before next batch...")
        time.sleep(wait_time)
    
    # Show final stats
    print("\n====== Collection Complete ======")
    print(f"Total processed: {total_processed}")
    print(f"Total successful: {successful_articles}")
    print(f"Total failed: {len(failed_urls)}")
    
    return all_articles, failed_urls

# Save articles to JSON
def save_to_json(articles, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(articles)} articles to '{filename}'")

# Run the script
if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting collection at {datetime.datetime.now()}")
    
    articles, failed_urls = collect_historical_news(SEARCH_QUERY, TARGET_ARTICLES)
    
    save_to_json(articles, OUTPUT_FILE)
    
    end_time = time.time()
    duration_mins = (end_time - start_time) / 60
    
    print(f"\nCompleted in {duration_mins:.2f} minutes")
    print(f"Collected {len(articles)} articles about '{SEARCH_QUERY}'")
