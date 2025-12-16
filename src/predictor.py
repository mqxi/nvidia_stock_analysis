import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class StockPredictor:
    def __init__(self):
        # Wir erhöhen die Anzahl der Bäume für mehr Stabilität
        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.features = []

    def prepare_data(self, df):
        """
        Bereitet die Daten vor.
        WICHTIG: Wir sagen jetzt die RENDITE (Returns) vorher, nicht den Preis!
        Das löst das Problem mit dem negativen R².
        """
        data = df.copy()

        # 1. Zielvariable: Die prozentuale Änderung von Morgen
        # shift(-1) schiebt die Daten um einen Tag zurück (Target)
        data['Target_Return'] = data['Close'].pct_change().shift(-1)
        
        # Features bereinigen (unendliche Werte entfernen)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        # Features definieren
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                         'SMA_20', 'SMA_50', 'RSI', 
                         'Bollinger_Upper', 'Bollinger_Lower', 'Daily_Return',
                         'MACD', 'MACD_Signal', 'ATR', 'OBV']
        
        # Nur Spalten nutzen, die wirklich da sind
        available_features = [f for f in self.features if f in data.columns]
        self.features = available_features
        
        X = data[self.features]
        y = data['Target_Return']
        
        return X, y

    def train(self, df):
        print("🧠 Trainiere Modell auf RELATIVER Rendite...")
        
        X, y = self.prepare_data(df)
        
        # Split (Zeitreihen-konform, nicht mischen!)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        
        # Evaluation
        predictions = self.model.predict(X_test)
        
        # Da wir nun kleine Prozentzahlen (z.B. 0.015) vorhersagen, ist der R² schwerer zu interpretieren.
        # Aber der Trend (Richtung) ist wichtiger.
        score = r2_score(y_test, predictions)
        
        # Richtungskorrektheit (Directional Accuracy) berechnen
        # Haben wir korrekt vorhergesagt, ob es hoch oder runter geht?
        correct_direction = np.sign(predictions) == np.sign(y_test)
        accuracy = np.mean(correct_direction) * 100
        
        print(f"✅ Training fertig.")
        print(f"   Richtungstrefferquote: {accuracy:.1f}% (Zufall wäre 50%)")
        print(f"   R² Score (Rendite): {score:.4f}")
        
        return self.model

    def predict_with_sentiment(self, df, sentiment_score=0):
        """
        Kombiniert technische Analyse (ML) mit News-Sentiment.
        
        Args:
            df: Der DataFrame mit den Aktienkursen.
            sentiment_score: Der Score aus der News-Analyse (-1 bis +1).
                             0 bedeutet Neutral (oder keine News).
        """
        # 1. Technische Vorhersage holen
        latest_data = df.iloc[[-1]][self.features]
        predicted_return = self.model.predict(latest_data)[0]
        
        current_price = df['Close'].iloc[-1]
        
        # 2. Sentiment-Einfluss berechnen (Heuristik)
        # Wir nehmen an: Sehr starke News (+1.0) können den Kurs um extra 1-2% bewegen.
        # Das ist ein einstellbarer Faktor ("Impact Factor").
        sentiment_impact = sentiment_score * 0.015 # 0.015 = max 1.5% Einfluss durch News
        
        # 3. Fusion: Technik + News
        final_predicted_return = predicted_return + sentiment_impact
        
        predicted_price = current_price * (1 + final_predicted_return)
        
        return {
            "current_price": current_price,
            "technical_return": predicted_return,
            "sentiment_impact": sentiment_impact,
            "final_predicted_return": final_predicted_return,
            "predicted_price": predicted_price
        }

# --- Test-Bereich ---
if __name__ == "__main__":
    from data_loader import load_stock_data
    from indicators import add_indicators
    
    # 1. Daten laden
    df = load_stock_data("NVDA", period="5y")
    df = add_indicators(df)
    
    # 2. Modell trainieren
    predictor = StockPredictor()
    predictor.train(df)
    
    # 3. Test: Wir simulieren mal News
    # Szenario A: Schlechte News (Sentiment -0.5)
    result = predictor.predict_with_sentiment(df, sentiment_score=-0.5)
    
    print("\n--- Vorhersage Analyse ---")
    print(f"Aktueller Preis:   ${result['current_price']:.2f}")
    print(f"KI (Rein Technisch): {result['technical_return']*100:.2f}%")
    print(f"News Einfluss:       {result['sentiment_impact']*100:.2f}%")
    print(f"Gesamt-Prognose:     {result['final_predicted_return']*100:.2f}%")
    print(f"Zielpreis Morgen:  ${result['predicted_price']:.2f}")