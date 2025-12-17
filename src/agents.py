import pandas as pd
import numpy as np

class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.vote = "NEUTRAL"
        self.reason = "Keine Daten verfügbar."
        self.confidence = 0.0

class TechnicalAgent(Agent):
    def analyze(self, df):
        last = df.iloc[-1]
        score = 0
        reasons = []

        # RSI
        if last['RSI'] < 30:
            score += 1
            reasons.append(f"RSI extrem niedrig ({last['RSI']:.0f}), Gegenbewegung wahrscheinlich.")
        elif last['RSI'] > 70:
            score -= 1
            reasons.append(f"RSI extrem hoch ({last['RSI']:.0f}), Überhitzung droht.")
        else:
            reasons.append(f"RSI ist neutral ({last['RSI']:.0f}).")
        
        # Bollinger
        if last['Close'] < last['Bollinger_Lower']:
            score += 1
            reasons.append("Preis unter unterem Bollinger Band (Kaufsignal).")
        elif last['Close'] > last['Bollinger_Upper']:
            score -= 1
            reasons.append("Preis über oberem Bollinger Band (Verkaufssignal).")

        # MACD
        if last['MACD'] > last['MACD_Signal']:
            score += 0.5
            reasons.append("MACD Trend ist positiv.")
        else:
            score -= 0.5
            reasons.append("MACD Trend ist negativ.")

        # Ergebnis
        if score >= 1:
            self.vote = "BULLISH"
            self.confidence = min(abs(score) / 3, 1.0)
        elif score <= -1:
            self.vote = "BEARISH"
            self.confidence = min(abs(score) / 3, 1.0)
        else:
            self.vote = "NEUTRAL"
            self.confidence = 0.5
            
        self.reason = " | ".join(reasons)
        return self

class SentimentAgent(Agent):
    def analyze(self, news_df):
        if news_df is None or news_df.empty:
            self.vote = "NEUTRAL"
            self.reason = "Keine aktuellen News-Daten gefunden."
            self.confidence = 0.0
            return self
            
        avg_sentiment = news_df['Sentiment_Score'].mean()
        
        # Social Media Filter
        social_df = news_df[news_df['Type'] == 'Social']
        social_sent = social_df['Sentiment_Score'].mean() if not social_df.empty else 0
        
        reasons = []
        score = 0
        
        # 1. Allgemeine Stimmung (News + Social)
        if avg_sentiment > 0.15:
            score += 1
            reasons.append(f"Gesamtstimmung ist positiv ({avg_sentiment:.2f}).")
        elif avg_sentiment < -0.15:
            score -= 1
            reasons.append(f"Gesamtstimmung ist negativ ({avg_sentiment:.2f}).")
        else:
            reasons.append(f"Gesamtstimmung ist neutral/ausgeglichen ({avg_sentiment:.2f}).")
            
        # 2. Social Media Hype Faktor
        if not social_df.empty:
            if social_sent > 0.25:
                score += 0.5
                reasons.append("Privatanleger (Social) sind euphorisch.")
            elif social_sent < -0.25:
                score -= 0.5
                reasons.append("Privatanleger (Social) haben Angst.")
            else:
                reasons.append("In Social Media herrscht Ruhe.")
        else:
            reasons.append("Keine Social Media Daten verfügbar.")

        # Ergebnis berechnen
        if score > 0.5:
            self.vote = "BULLISH"
        elif score < -0.5:
            self.vote = "BEARISH"
        else:
            self.vote = "NEUTRAL"
            
        # Falls aus irgendeinem Grund die reasons leer wären (sollte nicht passieren), Fallback:
        if not reasons:
            reasons.append("Datenlage ist uneindeutig.")

        self.reason = " | ".join(reasons)
        self.confidence = min(abs(avg_sentiment) * 3, 1.0) 
        return self

class QuantAgent(Agent):
    def analyze(self, prediction_dict, decomposition):
        pred_return = prediction_dict['final_predicted_return']
        
        reasons = []
        score = 0
        
        # ML Modell
        if pred_return > 0.005: 
            score += 1
            reasons.append(f"KI-Modell prognostiziert Anstieg (+{pred_return*100:.2f}%).")
        elif pred_return < -0.005:
            score -= 1
            reasons.append(f"KI-Modell prognostiziert Rückgang ({pred_return*100:.2f}%).")
        else:
            reasons.append(f"KI-Modell erwartet Seitwärtsbewegung ({pred_return*100:.2f}%).")
            
        # Saisonalität
        if decomposition is not None:
            # Check Trend der letzten Tage in der Saisonalität
            seasonal_curve = decomposition['seasonal'].iloc[-5:]
            if seasonal_curve.mean() > 0 and seasonal_curve.iloc[-1] > seasonal_curve.iloc[0]:
                score += 0.5
                reasons.append("Zyklisches Muster zeigt nach oben.")
            elif seasonal_curve.mean() < 0:
                score -= 0.5
                reasons.append("Zyklisches Muster zeigt nach unten.")
            else:
                reasons.append("Kein starker saisonaler Einfluss.")
        else:
            reasons.append("Zu wenig Daten für Zyklus-Analyse.")
        
        if score > 0.5:
            self.vote = "BULLISH"
        elif score < -0.5:
            self.vote = "BEARISH"
        else:
            self.vote = "NEUTRAL"
            
        self.reason = " | ".join(reasons)
        self.confidence = 0.8
        return self

class HedgeFund:
    def __init__(self):
        self.tech_agent = TechnicalAgent("Dr. Chart", "Technical Analysis")
        self.sent_agent = SentimentAgent("Mr. Hype", "Sentiment Analysis")
        self.quant_agent = QuantAgent("The Brain", "Quantitative Analysis")
        
    def get_verdict(self, df, news_df, prediction, decomposition):
        r1 = self.tech_agent.analyze(df)
        r2 = self.sent_agent.analyze(news_df)
        r3 = self.quant_agent.analyze(prediction, decomposition)
        
        agents = [r1, r2, r3]
        
        votes = [a.vote for a in agents]
        bullish = votes.count("BULLISH")
        bearish = votes.count("BEARISH")
        
        if bullish > bearish:
            return agents, "KAUFEN (BULLISH)", "green"
        elif bearish > bullish:
            return agents, "VERKAUFEN (BEARISH)", "red"
        else:
            return agents, "HALTEN (NEUTRAL)", "gray"