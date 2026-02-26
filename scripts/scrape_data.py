"""Simple, configurable scraper that extracts paragraph text from pages.

Usage: edit `urls.txt` (one URL per line) or pass a list of URLs to `scrape_pages(urls)`.

Notes:
- This is a lightweight helper for assembling raw text for experimentation.
- Respect robots.txt and site terms before scraping. Use sparingly.
"""

import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "conversations.txt"


def scrape_page(url: str, timeout: int = 10) -> str:
    headers = {"User-Agent": "AetheriaBot/0.1 (+https://example.com/)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""
    soup = BeautifulSoup(r.content, "html.parser")
    # extract visible paragraph text
    paragraphs = [p.get_text(separator=" ").strip() for p in soup.find_all("p")]
    # filter out very short paragraphs
    paragraphs = [p for p in paragraphs if len(p) > 40]
    return "\n\n".join(paragraphs)


def scrape_pages(urls: Iterable[str], out_path: Path = OUT_PATH, pause: float = 1.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for url in urls:
            print("Fetching:", url)
            text = scrape_page(url)
            if text:
                f.write(text)
                f.write("\n\n")
            time.sleep(pause)
    print("Saved raw scraped text to", out_path)


def urls_from_file(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == '__main__':
    # default behaviour: read urls.txt in repo root
    urls_file = Path(__file__).parents[1] / 'urls.txt'
    urls = urls_from_file(urls_file)
    if not urls:
        print("No URLs found in urls.txt. Add one URL per line to scrape pages.")
    else:
        scrape_pages(urls)