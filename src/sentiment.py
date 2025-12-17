import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# NLTK Ressourcen herunterladen (Caching)
for resource in ["vader_lexicon", "stopwords", "punkt"]:
    try:
        nltk.data.find(
            f"sentiment/{resource}.zip"
        ) if resource == "vader_lexicon" else nltk.data.find(f"corpora/{resource}.zip")
    except LookupError:
        nltk.download(resource, quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

        # --- STOPWORDS SETUP ---
        # 1. NLTK Listen für Englisch und Deutsch
        en_stops = set(stopwords.words("english"))
        de_stops = set(stopwords.words("german"))

        # 2. Finanz-spezifische Wörter, die die WordCloud verstopfen
        # (Wir wissen ja, dass es um Nvidia geht, das muss nicht riesig angezeigt werden)
        finance_stops = {
            "nvidia",
            "nvda",
            "stock",
            "stocks",
            "share",
            "shares",
            "market",
            "price",
            "forecast",
            "prediction",
            "analysis",
            "news",
            "update",
            "buy",
            "sell",
            "today",
            "now",
            "live",
            "watch",
            "video",
            "aktie",
            "aktien",
            "kurs",
            "prognose",
            "markt",
            "börse",
            "inc",
            "corp",
            "report",
            "results",
            "earning",
            "earnings",
        }

        # Alles kombinieren
        self.stop_words = en_stops.union(de_stops).union(finance_stops)

    def clean_text(self, text):
        """Entfernt Sonderzeichen und macht alles klein."""
        # Nur Buchstaben behalten
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", "", text)
        return text.lower()

    def get_text_for_wordcloud(self, news_df):
        """
        Gibt einen bereinigten String zurück, aus dem alle Stopwörter entfernt wurden.
        Ideal für die WordCloud Generierung.
        """
        if news_df is None or news_df.empty:
            return ""

        # Alle Titel zu einem langen String verbinden
        full_text = " ".join(news_df["Title"].astype(str))

        # Säubern
        cleaned_text = self.clean_text(full_text)

        # Stopwörter filtern
        words = cleaned_text.split()
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 2]

        return " ".join(filtered_words)

    def analyze_news(self, news_df):
        """
        Fügt Sentiment (VADER) und Subjektivität (TextBlob) hinzu.
        """
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        results = []
        for index, row in news_df.iterrows():
            title = str(row["Title"])

            # 1. VADER (Emotion)
            vader_score = self.sia.polarity_scores(title)["compound"]

            # 2. TextBlob (Fakt vs Meinung)
            blob = TextBlob(title)
            subjectivity = blob.sentiment.subjectivity

            results.append(
                {"Sentiment_Score": vader_score, "Subjectivity": subjectivity}
            )

        metrics_df = pd.DataFrame(results)
        # Indizes zurücksetzen für sauberen Concat
        news_df = news_df.reset_index(drop=True)
        metrics_df = metrics_df.reset_index(drop=True)

        return pd.concat([news_df, metrics_df], axis=1)
