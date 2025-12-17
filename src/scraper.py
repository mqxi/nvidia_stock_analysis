import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

class NewsScraper:
    def __init__(self):
        # Wir tarnen uns als normaler Browser, um nicht blockiert zu werden
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_nvidia_news(self, query="NVIDIA stock", max_items=20):
        """
        Crawlt News-Headlines via Google News RSS.
        
        Args:
            query (str): Suchbegriff (z.B. "NVIDIA", "Jensen Huang").
            max_items (int): Maximale Anzahl an News.
            
        Returns:
            pd.DataFrame: Tabelle mit Datum, Titel, Link und Quelle.
        """
        print(f"🕷️ Crawle News für: '{query}'...")
        
        # Google News RSS URL
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status() # Check auf HTTP Fehler
            
            # XML Parsen mit BeautifulSoup
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.findAll('item')
            
            news_list = []
            
            for index, item in enumerate(items):
                if index >= max_items:
                    break
                
                # Daten extrahieren
                title = item.title.text
                link = item.link.text
                pub_date_str = item.pubDate.text
                source = item.source.text if item.source else "Unknown"
                
                # Datum formatieren (Mo, 16 Dec 2024 10:00:00 GMT -> datetime)
                try:
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    pub_date = datetime.now() # Fallback

                news_list.append({
                    "Date": pub_date,
                    "Title": title,
                    "Source": source,
                    "Link": link
                })
            
            df = pd.DataFrame(news_list)
            print(f"✅ {len(df)} News-Artikel gefunden.")
            return df

        except Exception as e:
            print(f"❌ Fehler beim Scraping: {e}")
            return pd.DataFrame()

    def get_stocktwits_feed(self, symbol="NVDA"):
        """
        Holt echte Trader-Kommentare von Stocktwits (Twitter-Alternative für Finanzen).
        Dies ist die beste Quelle für kurzfristige 'Insider'-Stimmung.
        """
        print(f"🐦 Hole Stocktwits für {symbol}...")
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        
        try:
            r = requests.get(url, headers=self.headers)
            data = r.json()
            
            messages = []
            for msg in data['messages']:
                # Wir filtern Spam raus
                body = msg['body']
                user = msg['user']['username']
                time_str = msg['created_at'] # Format: 2024-12-16T15:30:00Z
                
                # Sentiment Labels (Bullish/Bearish) wenn vorhanden
                sentiment_label = msg['entities']['sentiment']['basic'] if msg['entities']['sentiment'] else "Neutral"
                
                messages.append({
                    "Date": datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ"),
                    "Title": f"@{user}: {body}", # Wir nutzen 'Title' damit der SentimentAnalyzer es versteht
                    "Source": "Stocktwits",
                    "Type": "Social",
                    "Label": sentiment_label # Zusatzinfo für uns
                })
            
            return pd.DataFrame(messages)
        except Exception as e:
            print(f"❌ Fehler Stocktwits: {e}")
            return pd.DataFrame()

    def get_reddit_posts(self, subreddit="nvidia", limit=15):
        """Holt die neuesten Diskussionen von Reddit (JSON Trick)."""
        print(f"👽 Crawle r/{subreddit}...")
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
        
        try:
            r = requests.get(url, headers=self.headers)
            data = r.json()
            
            posts = []
            for child in data['data']['children']:
                post = child['data']
                title = post['title']
                text = post['selftext'][:200] # Nur die ersten 200 Zeichen
                full_text = f"{title} - {text}"
                
                # Unix Timestamp umwandeln
                dt = datetime.fromtimestamp(post['created_utc'])
                
                posts.append({
                    "Date": dt,
                    "Title": full_text,
                    "Source": f"Reddit r/{subreddit}",
                    "Type": "Social"
                })
            return pd.DataFrame(posts)
        except Exception as e:
            print(f"❌ Fehler Reddit: {e}")
            return pd.DataFrame()

    def get_all_sources(self, ticker="NVDA"):
        """Aggregiert ALLES: News + Stocktwits + Reddit"""
        df_news = self.get_nvidia_news(f"{ticker} stock")
        df_st = self.get_stocktwits_feed(ticker)
        df_reddit = self.get_reddit_posts("wallstreetbets", limit=10) # WSB ist wichtig für Hypes
        
        # Alle DataFrames zusammenkleben
        full_df = pd.concat([df_news, df_st, df_reddit], ignore_index=True)
        
        # Nach Datum sortieren (neueste zuerst)
        if not full_df.empty:
            full_df = full_df.sort_values(by="Date", ascending=False)
            
        return full_df

    def get_earnings_calendar(self, ticker_symbol="NVDA"):
        """
        Holt geplante Events (Earnings Calls) via yfinance.
        Dies ist zuverlässiger als manuelles Scraping von Kalender-Seiten.
        """
        import yfinance as yf
        try:
            ticker = yf.Ticker(ticker_symbol)
            calendar = ticker.calendar
            
            # yfinance ändert die Struktur oft, wir fangen das ab
            if calendar is None or (isinstance(calendar, dict) and not calendar):
                print("⚠️ Keine Kalender-Daten gefunden.")
                return None
            
            # Wenn es ein Dictionary ist (neue yfinance Version)
            if isinstance(calendar, dict):
                return pd.DataFrame(calendar).T
            
            # Wenn es schon ein DataFrame ist
            return calendar
            
        except Exception as e:
            print(f"❌ Fehler beim Kalender-Abruf: {e}")
            return None

# --- Test-Bereich ---
if __name__ == "__main__":
    scraper = NewsScraper()
    
    # 1. Test: News Crawlen
    news_df = scraper.get_nvidia_news(query="NVIDIA Jensen Huang AI")
    if not news_df.empty:
        print("\n--- Letzte Headlines ---")
        print(news_df[['Date', 'Title']].head())
        # Speichern für spätere Analysen
        news_df.to_csv("data/nvidia_news_raw.csv", index=False)

    # 2. Test: Events
    print("\n--- Nächste Events ---")
    events = scraper.get_earnings_calendar()
    if events is not None:
        print(events)