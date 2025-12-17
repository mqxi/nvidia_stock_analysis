import random
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


class NewsScraper:
    def __init__(self):
        # Wir rotieren User-Agents, um weniger wie ein Bot zu wirken
        self.session = requests.Session()

    def _get_headers(self):
        """
        Simuliert einen kompletten Browser-Header, um 403 Fehler zu umgehen.
        """
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]

        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://stocktwits.com/",
            "Origin": "https://stocktwits.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Connection": "keep-alive",
        }

    def get_nvidia_news(self, query="NVIDIA stock", max_items=200):
        """Google News RSS"""
        print(f"🕷️ Crawle Google News: '{query}'...")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            soup = BeautifulSoup(response.content, features="lxml-xml")
            items = soup.findAll("item")

            news_list = []
            for item in items[:max_items]:
                try:
                    pub_date = datetime.strptime(
                        item.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except ValueError:
                    pub_date = datetime.now()

                news_list.append(
                    {
                        "Date": pub_date,
                        "Title": item.title.text,
                        "Source": item.source.text if item.source else "GoogleNews",
                        "Type": "News",
                    }
                )
            return pd.DataFrame(news_list)
        except Exception as e:
            print(f"❌ Fehler Google News: {e}")
            return pd.DataFrame()

    def get_stocktwits_feed(self, symbol="NVDA"):
        """Stocktwits API mit maximaler Tarnung"""
        print(f"🐦 Hole Stocktwits für {symbol}...")
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

        try:
            # Wir nutzen die Session und die vollen Header
            r = self.session.get(url, headers=self._get_headers(), timeout=5)

            if r.status_code == 403:
                print("⚠️ Stocktwits Block (403). Versuche Reddit als Fallback...")
                return (
                    pd.DataFrame()
                )  # Leeres DF zurückgeben, damit der Code weiterläuft

            if r.status_code != 200:
                print(f"⚠️ Stocktwits Status: {r.status_code}")
                return pd.DataFrame()

            data = r.json()
            messages = []
            for msg in data.get("messages", []):
                body = msg["body"]
                user = msg["user"]["username"]
                time_str = msg["created_at"]

                sentiment_label = "Neutral"
                if msg.get("entities") and msg["entities"].get("sentiment"):
                    sentiment_label = msg["entities"]["sentiment"]["basic"]

                dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

                messages.append(
                    {
                        "Date": dt,
                        "Title": f"@{user}: {body}",
                        "Source": "Stocktwits",
                        "Type": "Social",
                        "Label": sentiment_label,
                    }
                )

            return pd.DataFrame(messages)

        except Exception as e:
            print(f"❌ Fehler Stocktwits: {e}")
            return pd.DataFrame()

    def get_reddit_posts(self, subreddit="nvidia", limit=200):
        """Reddit JSON"""
        print(f"👽 Crawle r/{subreddit}...")
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"

        try:
            # Reddit braucht einen sehr spezifischen User-Agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            r = requests.get(url, headers=headers, timeout=5)
            data = r.json()

            posts = []
            if "data" in data and "children" in data["data"]:
                for child in data["data"]["children"]:
                    post = child["data"]
                    title = post["title"]
                    text = post.get("selftext", "")[:200]
                    full_text = f"{title} - {text}"

                    dt = datetime.fromtimestamp(post["created_utc"])

                    posts.append(
                        {
                            "Date": dt,
                            "Title": full_text,
                            "Source": f"Reddit r/{subreddit}",
                            "Type": "Social",
                        }
                    )
            return pd.DataFrame(posts)
        except Exception as e:
            print(f"❌ Fehler Reddit: {e}")
            return pd.DataFrame()

    def get_all_sources(self, ticker="NVDA"):
        # Parallel holen
        df_news = self.get_nvidia_news(f"{ticker} stock")
        df_st = self.get_stocktwits_feed(ticker)

        # Reddit holen (WallStreetBets & Nvidia Subreddit)
        df_reddit_wsb = self.get_reddit_posts("wallstreetbets", limit=200)
        df_reddit_nvda = self.get_reddit_posts("nvidia", limit=200)

        dfs = [
            d for d in [df_news, df_st, df_reddit_wsb, df_reddit_nvda] if not d.empty
        ]

        if not dfs:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Title",
                    "Source",
                    "Type",
                    "Sentiment_Score",
                    "Subjectivity",
                ]
            )

        full_df = pd.concat(dfs, ignore_index=True)

        # Duplikate entfernen (manchmal posten Leute das Gleiche)
        full_df.drop_duplicates(subset=["Title"], inplace=True)

        if "Date" in full_df.columns:
            full_df = full_df.sort_values(by="Date", ascending=False)

        return full_df
