# -*- coding: cp1251 -*-
import requests
from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer
import re 

# 1. Define the song links
song_links = [
    "https://genius.com/Kanye-west-every-hour-lyrics",
    "https://genius.com/Kanye-west-selah-lyrics",
    "https://genius.com/Kanye-west-follow-god-lyrics",
    "https://genius.com/Kanye-west-closed-on-sunday-lyrics",
    "https://genius.com/Kanye-west-on-god-lyrics",
    "https://genius.com/Kanye-west-everything-we-need-lyrics",
    "https://genius.com/Kanye-west-water-lyrics",
    "https://genius.com/Kanye-west-god-is-lyrics",
    "https://genius.com/Kanye-west-hands-on-lyrics",
    "https://genius.com/Kanye-west-use-this-gospel-lyrics",
    "https://genius.com/Kanye-west-jesus-is-lord-lyrics",
    "https://genius.com/Kanye-west-praise-god-lyrics",
    "https://genius.com/Kanye-west-god-breathed-lyrics"
]


def fetch_lyrics(url):
    """Fetch lyrics from a Genius.com URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract song title
        title = soup.find('h1').text.strip()
        
        # New method to extract lyrics
        lyrics_div = soup.find('div', {'data-lyrics-container': 'true'})
        if not lyrics_div:
            lyrics_div = soup.find('div', class_='Lyrics__Container-sc-1ynbvzw-6 YYrds')
        
        if lyrics_div:
            # Clean up the lyrics
            lyrics = lyrics_div.get_text('\n')
            lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse] tags
            lyrics = re.sub(r'\n{2,}', '\n\n', lyrics)  # Remove excessive newlines
            return f"{title}\n\n{lyrics.strip()}\n\n{'='*50}\n"
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_all_lyrics(links, filename="kanye_lyrics.txt"):
    """Save all lyrics to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, link in enumerate(links, 1):
            print(f"Fetching song {i}/{len(links)}...")
            lyrics = fetch_lyrics(link)
            if lyrics:
                f.write(lyrics)
                print(f"Saved: {lyrics.splitlines()[0]}")
            else:
                print(f"Failed to fetch: {link}")
        
    print(f"\nAll lyrics saved to {filename}")


# Execute the scraping and saving
save_all_lyrics(song_links)


# 2. Function to fetch lyrics
def get_lyrics(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1').text.strip()
        lyrics_div = soup.find('div', class_=lambda x: x and 'Lyrics__Container' in x)
        lyrics = '\n'.join([p.text for p in lyrics_div.find_all('p')]) if lyrics_div else ""
        return f"{title}\n{lyrics}"
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

