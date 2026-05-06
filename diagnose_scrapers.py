import asyncio
import feedparser

FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "NYT": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "CNN": "http://rss.cnn.com/rss/cnn_world.rss",
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

for name, url in FEEDS.items():
    f = feedparser.parse(url)
    if not f.entries:
        print(f"{name}: 0 entries — feed dead")
        continue
    sample = f.entries[0]
    has_enclosure = bool(getattr(sample, 'enclosures', None))
    has_media_thumb = 'media_thumbnail' in sample
    has_media_content = 'media_content' in sample
    print(f"{name}: {len(f.entries)} entries | enclosure={has_enclosure} thumb={has_media_thumb} content={has_media_content}")