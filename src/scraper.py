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