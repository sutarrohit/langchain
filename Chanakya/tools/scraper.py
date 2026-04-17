import trafilatura
from curl_cffi.requests import Session
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


class Scraper:
    def __init__(self):
        self.session = Session(impersonate="chrome")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def scrape(self, url: str) -> dict:
        """
        Fetches and extracts clean text from a URL.
        Uses requests + trafilatura to handle bot detection.
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            downloaded = response.text

            if not downloaded:
                return {"success": False, "error": "Empty response from URL"}

            text = trafilatura.extract(
                downloaded,
                with_metadata=True,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )

            if not text:
                return {
                    "success": False,
                    "error": "No article content found — page may be paywalled or JS-only",
                }

            return {
                "success": True,
                "url": url,
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Create the tool
@tool
def scrape_article(url: str) -> dict:
    """Scrape full article content from a URL.

    Useful for extracting clean text content from web articles.
    Returns the full text, word count, and metadata.

    Args:
        url: The URL of the article to scrape

    Returns:
        Dict containing success status, extracted text, word_count, char_count, and url
    """
    scraper = Scraper()
    return scraper.scrape(url)
