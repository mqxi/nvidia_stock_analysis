import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
from collections import Counter
import re

# NLTK Lexicon herunterladen (passiert nur beim ersten Mal)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        """Entfernt Sonderzeichen für bessere WordClouds."""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()

    def analyze_news(self, news_df):
        """
        Erweitert: Nutzt VADER (Polarität) UND TextBlob (Subjektivität).
        """
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        print("🧠 Analysiere Sentiment & Subjektivität...")
        
        results = []
        for index, row in news_df.iterrows():
            title = row['Title']
            
            # 1. VADER (Gut für Emotionen in Short-Text)
            vader_score = self.sia.polarity_scores(title)['compound']
            
            # 2. TextBlob (Gut für Subjektivität: 0=Fakt, 1=Meinung)
            blob = TextBlob(title)
            subjectivity = blob.sentiment.subjectivity
            
            results.append({
                'Sentiment_Score': vader_score,
                'Subjectivity': subjectivity
            })

        # Ergebnisse an den DataFrame hängen
        metrics_df = pd.DataFrame(results)
        news_df = pd.concat([news_df.reset_index(drop=True), metrics_df], axis=1)
        
        return news_df

    def get_top_keywords(self, news_df, top_n=10):
        """Extrahiert die häufigsten Wörter für die WordCloud."""
        if news_df.empty:
            return {}
            
        all_text = " ".join(news_df['Title'].dropna())
        cleaned_text = self.clean_text(all_text)
        
        # Stopwords entfernen (einfache Liste)
        stopwords = set(['to', 'the', 'of', 'in', 'for', 'on', 'and', 'a', 'is', 'at', 'stock', 'nvidia', 'nvda', 'shares', 'market'])
        words = [w for w in cleaned_text.split() if w not in stopwords and len(w) > 2]
        
        return dict(Counter(words).most_common(top_n))