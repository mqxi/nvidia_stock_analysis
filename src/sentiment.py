import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# NLTK Lexicon herunterladen (passiert nur beim ersten Mal)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_news(self, news_df):
        """
        Nimmt den News-DataFrame und fügt einen 'Sentiment_Score' hinzu.
        Score: -1 (sehr negativ) bis +1 (sehr positiv).
        """
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        print("🧠 Analysiere Sentiment der Headlines...")
        
        # Wir berechnen den 'compound' Score für jeden Titel
        # Lambda-Funktion wendet den Analyzer auf jede Zeile an
        news_df['Sentiment_Score'] = news_df['Title'].apply(
            lambda title: self.sia.polarity_scores(title)['compound']
        )
        
        return news_df

    def get_daily_sentiment(self, news_df):
        """
        Aggregiert die News pro Tag zu einem Durchschnittswert.
        Wichtig, um es mit den Aktiendaten (1 Zeile pro Tag) zu mergen.
        """
        # Nur Datum (ohne Uhrzeit) für Gruppierung nutzen
        news_df['Date_Only'] = pd.to_datetime(news_df['Date']).dt.date
        
        daily_sentiment = news_df.groupby('Date_Only')['Sentiment_Score'].mean().reset_index()
        daily_sentiment.columns = ['Date', 'Sentiment_Avg']
        
        # Index wieder auf Datetime setzen für den Merge später
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        
        return daily_sentiment

# --- Test ---
if __name__ == "__main__":
    # Wir simulieren kurz News, falls der Scraper gerade nichts liefert
    data = {
        'Date': ['2024-12-16', '2024-12-16', '2024-12-17'],
        'Title': [
            "NVIDIA reaches new all-time high!", 
            "Investors are worried about regulations.", 
            "Jensen Huang announces revolutionary AI chip."
        ]
    }
    df = pd.DataFrame(data)
    
    analyzer = SentimentAnalyzer()
    scored_df = analyzer.analyze_news(df)
    
    print("\n--- Einzelne Bewertungen ---")
    print(scored_df[['Title', 'Sentiment_Score']])
    
    print("\n--- Tages-Durchschnitt ---")
    daily = analyzer.get_daily_sentiment(scored_df)
    print(daily)