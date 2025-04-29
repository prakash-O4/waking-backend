import json
import requests
import sqlite3
import time
import os
from typing import List, Optional
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from pydantic import BaseModel, HttpUrl
from contextlib import contextmanager

# Pydantic models for request/response
class ScrapingRequest(BaseModel):
    url: HttpUrl
    force_update: bool = False

class VideoRequest(BaseModel):
    article_url: HttpUrl

class ArticleResponse(BaseModel):
    id: Optional[int] = None
    title: str
    url: str
    image_url: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    categories: List[str] = []
    tags: List[str] = []
    videos: List[str] = []

class ScrapingResponse(BaseModel):
    message: str
    articles_processed: int
    articles_saved: int
    videos_saved: int

# Initialize FastAPI app
app = FastAPI(
    title="Football Highlights Scraper API",
    description="API to scrape football highlights from websites and store them in a SQLite database",
    version="1.0.0"
)

# Database connection manager
@contextmanager
def get_db_connection():
    conn = sqlite3.connect('football_highlights.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    try:
        yield conn
    finally:
        conn.close()

# Create database tables
def create_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create articles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            image_url TEXT,
            date TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create categories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            name TEXT,
            FOREIGN KEY (article_id) REFERENCES articles (id),
            UNIQUE(article_id, name)
        )
        ''')
        
        # Create tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            name TEXT,
            FOREIGN KEY (article_id) REFERENCES articles (id),
            UNIQUE(article_id, name)
        )
        ''')
        
        # Create videos table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            video_url TEXT,
            FOREIGN KEY (article_id) REFERENCES articles (id),
            UNIQUE(article_id, video_url)
        )
        ''')
        
        conn.commit()

# Initialize database at startup
@app.on_event("startup")
def startup_event():
    create_database()

# Extract articles from URL
def extract_articles_from_url(url):
    # Fetch the page content
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        html_content = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    
    # Parse the content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all article elements
    articles = soup.find_all('article')
    
    articles_data = []
    
    for article in articles:
        # Extract data for each article
        article_data = {}
        
        # Title
        title_element = article.select_one('.read-title h4 a')
        if title_element:
            article_data['title'] = title_element.text.strip()
            article_data['url'] = title_element.get('href', '')
        
        # Image URL
        img_element = article.select_one('.read-img img')
        if img_element:
            article_data['image_url'] = img_element.get('src', '')
        
        # Date
        date_element = article.select_one('.posts-date a')
        if date_element:
            article_data['date'] = date_element.text.strip()
        
        # Categories
        categories = []
        cat_elements = article.select('.read-categories .cat-links a')
        for cat in cat_elements:
            categories.append(cat.text.strip())
        article_data['categories'] = categories
        
        # Tags from article classes
        tags = []
        article_classes = article.get('class', [])
        for cls in article_classes:
            if cls.startswith('tag-'):
                tags.append(cls.replace('tag-', ''))
        article_data['tags'] = tags
        
        # Description
        desc_element = article.select_one('.post-description')
        if desc_element:
            # Get text and remove the "Watch Video" if present
            desc_text = desc_element.get_text(strip=True)
            article_data['description'] = desc_text.replace('Watch Video', '').strip()
        
        articles_data.append(article_data)

    return articles_data

# Extract videos from article
def extract_video_urls_from_article(url):
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all video tags
        video_tags = soup.find_all('video')
        
        # Extract source URLs from video tags
        video_urls = []
        for video in video_tags:
            source_tags = video.find_all('source')
            for source in source_tags:
                video_url = source.get('src')
                if video_url:
                    print("video url ",video_url)
                    video_urls.append(video_url)
        
        return video_urls
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching video URL: {str(e)}")

# Save articles to database
def save_to_database(articles_data, force_update=False):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        articles_saved = 0
        videos_saved = 0
        
        for article in articles_data:
            try:
                # Insert article
                cursor.execute('''
                INSERT OR IGNORE INTO articles (title, url, image_url, date, description)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    article.get('title', ''),
                    article.get('url', ''),
                    article.get('image_url', ''),
                    article.get('date', ''),
                    article.get('description', '')
                ))
                
                # Get article ID (either the new one or existing one)
                is_new = cursor.rowcount > 0
                if is_new:
                    articles_saved += 1
                    article_id = cursor.lastrowid
                else:
                    # Get ID of the existing article
                    cursor.execute("SELECT id FROM articles WHERE url = ?", (article.get('url', ''),))
                    article_id = cursor.fetchone()[0]
                
                # Insert categories
                for category in article.get('categories', []):
                    cursor.execute('''
                    INSERT OR IGNORE INTO categories (article_id, name)
                    VALUES (?, ?)
                    ''', (article_id, category))
                
                # Insert tags
                for tag in article.get('tags', []):
                    cursor.execute('''
                    INSERT OR IGNORE INTO tags (article_id, name)
                    VALUES (?, ?)
                    ''', (article_id, tag))
                
                # Fetch and insert videos if article was newly added or we're forcing update
                if is_new or force_update:
                    article_url = article.get('url', '')
                    if article_url:
                        # Add a delay to be respectful to the server
                        time.sleep(1)
                        
                        # Extract videos from the article page
                        video_urls = extract_video_urls_from_article(article_url)
                        
                        # If force update, remove existing videos
                        if force_update and not is_new:
                            cursor.execute('DELETE FROM videos WHERE article_id = ?', (article_id,))
                        
                        # Insert videos
                        for video_url in video_urls:
                            cursor.execute('''
                            INSERT OR IGNORE INTO videos (article_id, video_url)
                            VALUES (?, ?)
                            ''', (article_id, video_url))
                            
                            if cursor.rowcount > 0:
                                videos_saved += 1
                
                conn.commit()
                
            except sqlite3.Error as e:
                print(f"Database error for article {article.get('title', '')}: {e}")
                conn.rollback()
            
            except Exception as e:
                print(f"Error processing article {article.get('title', '')}: {e}")
                conn.rollback()
        
        return len(articles_data), articles_saved, videos_saved

# Background task to process scraping requests
def process_scraping(url: str, force_update: bool = False):
    try:
        articles_data = extract_articles_from_url(url)
        articles_processed, articles_saved, videos_saved = save_to_database(articles_data, force_update)
        return {
            "articles_processed": articles_processed,
            "articles_saved": articles_saved,
            "videos_saved": videos_saved
        }
    except Exception as e:
        print(f"Error in background task: {e}")
        return {
            "articles_processed": 0,
            "articles_saved": 0,
            "videos_saved": 0,
            "error": str(e)
        }

# Get all articles from the database
def get_all_articles():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            a.id, a.title, a.url, a.image_url, a.date, a.description
        FROM 
            articles a
        ORDER BY 
            a.id ASC
        ''')
        
        articles = []
        for row in cursor.fetchall():
            article = {
                'id': row['id'],
                'title': row['title'],
                'url': row['url'],
                'image_url': row['image_url'],
                'date': row['date'],
                'description': row['description'],
                'categories': [],
                'tags': [],
                'videos': []
            }
            
            # Get categories for this article
            cursor.execute('SELECT name FROM categories WHERE article_id = ?', (article['id'],))
            article['categories'] = [cat[0] for cat in cursor.fetchall()]
            
            # Get tags for this article
            cursor.execute('SELECT name FROM tags WHERE article_id = ?', (article['id'],))
            article['tags'] = [tag[0] for tag in cursor.fetchall()]
            
            # Get videos for this article
            cursor.execute('SELECT video_url FROM videos WHERE article_id = ?', (article['id'],))
            article['videos'] = [video[0] for video in cursor.fetchall()]
            
            articles.append(article)
        
        return articles

# Get single article by ID
def get_article(article_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            a.id, a.title, a.url, a.image_url, a.date, a.description
        FROM 
            articles a
        WHERE
            a.id = ?
        ''', (article_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        article = {
            'id': row['id'],
            'title': row['title'],
            'url': row['url'],
            'image_url': row['image_url'],
            'date': row['date'],
            'description': row['description'],
            'categories': [],
            'tags': [],
            'videos': []
        }
        
        # Get categories for this article
        cursor.execute('SELECT name FROM categories WHERE article_id = ?', (article['id'],))
        article['categories'] = [cat[0] for cat in cursor.fetchall()]
        
        # Get tags for this article
        cursor.execute('SELECT name FROM tags WHERE article_id = ?', (article['id'],))
        article['tags'] = [tag[0] for tag in cursor.fetchall()]
        
        # Get videos for this article
        cursor.execute('SELECT video_url FROM videos WHERE article_id = ?', (article['id'],))
        article['videos'] = [video[0] for video in cursor.fetchall()]
        
        return article

# API ENDPOINTS

@app.get("/")
async def root():
    return {"message": "Football Highlights Scraper API"}

@app.post("/scrape", response_model=ScrapingResponse)
async def scrape_url(request: ScrapingRequest, background_tasks: BackgroundTasks):
    """
    Scrape football highlights from the provided URL and store in the database.
    If force_update is True, it will update existing articles' videos.
    """
    # Start the scraping process in the background
    result = process_scraping(str(request.url), request.force_update)
    
    return {
        "message": "Scraping completed successfully",
        "articles_processed": result["articles_processed"],
        "articles_saved": result["articles_saved"],
        "videos_saved": result["videos_saved"]
    }

@app.post("/extract-videos", response_model=List[str])
async def extract_videos(request: VideoRequest):
    """
    Extract videos from an article URL without saving to the database.
    """
    video_urls = extract_video_urls_from_article(str(request.article_url))
    return video_urls

@app.get("/articles", response_model=List[ArticleResponse])
async def get_articles(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    """
    Get all articles from the database with pagination.
    """
    articles = get_all_articles()
    return articles

@app.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article_by_id(article_id: int):
    """
    Get a single article by ID.
    """
    article = get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article with ID {article_id} not found")
    return article

@app.get("/search", response_model=List[ArticleResponse])
async def search_articles(
    q: str = Query(None, description="Search term in title or description"),
    category: str = Query(None, description="Filter by category"),
    tag: str = Query(None, description="Filter by tag")
):
    """
    Search articles by title, description, category, or tag.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Base query parts
        select_part = """
        SELECT DISTINCT a.id FROM articles a
        """
        where_conditions = []
        params = []
        
        # Add joins and conditions based on filters
        if category:
            select_part += "LEFT JOIN categories c ON a.id = c.article_id "
            where_conditions.append("c.name LIKE ?")
            params.append(f"%{category}%")
        
        if tag:
            select_part += "LEFT JOIN tags t ON a.id = t.article_id "
            where_conditions.append("t.name LIKE ?")
            params.append(f"%{tag}%")
        
        if q:
            where_conditions.append("(a.title LIKE ? OR a.description LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        
        # Construct the final query
        query = select_part
        if where_conditions:
            query += "WHERE " + " AND ".join(where_conditions)
        
        # Execute the query to get matching article IDs
        cursor.execute(query, params)
        article_ids = [row[0] for row in cursor.fetchall()]
        
        # Get full article data for each ID
        articles = []
        for article_id in article_ids:
            article = get_article(article_id)
            if article:
                articles.append(article)
        
        return articles

@app.delete("/articles/{article_id}")
async def delete_article(article_id: int):
    """
    Delete an article and all its related data.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if article exists
        cursor.execute("SELECT id FROM articles WHERE id = ?", (article_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Article with ID {article_id} not found")
        
        try:
            # Delete related data first (foreign key constraints)
            cursor.execute("DELETE FROM videos WHERE article_id = ?", (article_id,))
            cursor.execute("DELETE FROM categories WHERE article_id = ?", (article_id,))
            cursor.execute("DELETE FROM tags WHERE article_id = ?", (article_id,))
            
            # Delete the article
            cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
            
            conn.commit()
            return {"message": f"Article with ID {article_id} and all related data deleted successfully"}
        
        except sqlite3.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/export")
async def export_data():
    """
    Export all data from the database to a JSON file.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                a.id, a.title, a.url, a.image_url, a.date, a.description, a.created_at
            FROM 
                articles a
            ORDER BY 
                a.id
            ''')
            
            rows = cursor.fetchall()
            articles = []
            
            for row in rows:
                article = {
                    'id': row['id'],
                    'title': row['title'],
                    'url': row['url'],
                    'image_url': row['image_url'],
                    'date': row['date'],
                    'description': row['description'],
                    'created_at': row['created_at'],
                    'categories': [],
                    'tags': [],
                    'videos': []
                }
                
                # Get categories for this article
                cursor.execute('SELECT name FROM categories WHERE article_id = ?', (article['id'],))
                article['categories'] = [cat[0] for cat in cursor.fetchall()]
                
                # Get tags for this article
                cursor.execute('SELECT name FROM tags WHERE article_id = ?', (article['id'],))
                article['tags'] = [tag[0] for tag in cursor.fetchall()]
                
                # Get videos for this article
                cursor.execute('SELECT video_url FROM videos WHERE article_id = ?', (article['id'],))
                article['videos'] = [video[0] for video in cursor.fetchall()]
                
                articles.append(article)
            
            # Write to JSON file
            output_file = 'football_highlights_export.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            
            return {"message": f"Data successfully exported to {output_file}", "articles_count": len(articles)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get database statistics.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Articles count
        cursor.execute("SELECT COUNT(*) FROM articles")
        articles_count = cursor.fetchone()[0]
        
        # Videos count
        cursor.execute("SELECT COUNT(*) FROM videos")
        videos_count = cursor.fetchone()[0]
        
        # Categories count
        cursor.execute("SELECT COUNT(DISTINCT name) FROM categories")
        categories_count = cursor.fetchone()[0]
        
        # Tags count
        cursor.execute("SELECT COUNT(DISTINCT name) FROM tags")
        tags_count = cursor.fetchone()[0]
        
        # Most popular categories
        cursor.execute("""
        SELECT name, COUNT(*) as count
        FROM categories
        GROUP BY name
        ORDER BY count DESC
        LIMIT 5
        """)
        top_categories = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Articles with most videos
        cursor.execute("""
        SELECT a.id, a.title, COUNT(v.id) as video_count
        FROM articles a
        JOIN videos v ON a.id = v.article_id
        GROUP BY a.id
        ORDER BY video_count DESC
        LIMIT 5
        """)
        top_articles = [{"id": row[0], "title": row[1], "video_count": row[2]} for row in cursor.fetchall()]
        
        return {
            "total_articles": articles_count,
            "total_videos": videos_count,
            "unique_categories": categories_count,
            "unique_tags": tags_count,
            "top_categories": top_categories,
            "top_articles_by_videos": top_articles
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)