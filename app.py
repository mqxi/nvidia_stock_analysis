import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Importiere unsere eigenen Module
from src.data_loader import load_stock_data
from src.indicators import add_indicators
from src.scraper import NewsScraper
from src.sentiment import SentimentAnalyzer
from src.predictor import StockPredictor

st.set_page_config(page_title="NVIDIA Stock AI", layout="wide", page_icon="📈")

st.title("🚀 NVIDIA (NVDA) Deep Dive Dashboard")
st.markdown("Erweiterte statistische Analyse & KI-Prognose.")

# --- Sidebar ---
st.sidebar.header("Konfiguration")
ticker = st.sidebar.text_input("Aktien Ticker", "NVDA")
period = st.sidebar.selectbox("Zeitraum", ["6mo", "1y", "2y", "5y"], index=1)

if st.sidebar.button("Daten aktualisieren 🔄"):
    st.cache_data.clear()

# --- Cache Funktionen ---
@st.cache_data
def get_data(ticker, period):
    df = load_stock_data(ticker, period=period)
    df = add_indicators(df)
    return df

@st.cache_data
def get_news_and_sentiment(ticker):
    scraper = NewsScraper()
    news_df = scraper.get_nvidia_news(query=f"{ticker} stock news", max_items=10)
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_news(news_df)

@st.cache_resource
def train_model(df):
    predictor = StockPredictor()
    predictor.train(df)
    return predictor

# --- Hauptlogik ---
with st.spinner('Lade komplexe Finanzdaten...'):
    df = get_data(ticker, period)

if df is None:
    st.error("Daten konnten nicht geladen werden.")
    st.stop()

# Key Metrics oben
latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Kurs", f"${latest['Close']:.2f}", f"{latest['Daily_Return']*100:.2f}%")
col2.metric("RSI (Momentum)", f"{latest['RSI']:.0f}", "Neutral" if 30 < latest['RSI'] < 70 else "Extrem")
col3.metric("ATR (Volatilität)", f"${latest['ATR']:.2f}", "Schwankungsbreite")
col4.metric("MACD Signal", "KAUFEN" if latest['MACD'] > latest['MACD_Signal'] else "VERKAUFEN", delta_color="normal")

# --- Tabs für bessere Übersicht ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart & Trend", "📉 Momentum & MACD", "🌊 Volumen (OBV)", "🧠 KI Prognose"])

# TAB 1: Hauptchart
with tab1:
    st.subheader("Preisentwicklung & Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='OHLC'))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=2), name='SMA 50'))
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ Erklärung: Was sehen wir hier?"):
        st.markdown("""
        * **Bollinger Bands (Grau):** Zeigen die Standardabweichung. Wenn der Kurs das obere Band berührt, ist er statistisch "teuer". Wenn sich die Bänder zusammenziehen ("Squeeze"), steht oft ein Ausbruch bevor.
        * **SMA 50 (Orange):** Der Durchschnitt der letzten 50 Tage. Dient oft als starke Unterstützungslinie bei Aufwärtstrends.
        """)

# TAB 2: MACD & RSI
with tab2:
    st.subheader("MACD Trend Analyse")
    
    
    # MACD Plot
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    
    # MACD Line & Signal
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange'), name='Signal'), row=1, col=1)
    
    # Histogramm
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='Histogramm'), row=2, col=1)
    
    fig_macd.update_layout(height=500)
    st.plotly_chart(fig_macd, use_container_width=True)
    
    st.info("""
    **MACD Erklärung:**
    * **MACD > Signal (Blaue Linie über Orange):** Bullish (Aufwärtstrend) 📈
    * **MACD < Signal (Blaue Linie unter Orange):** Bearish (Abwärtstrend) 📉
    * Das Histogramm zeigt die Stärke des Trends.
    """)

# TAB 3: Volumen & OBV
with tab3:
    st.subheader("On-Balance Volume (OBV)")
    
    fig_obv = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_obv.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Preis'), row=1, col=1)
    fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='purple'), name='OBV'), row=2, col=1)
    
    st.plotly_chart(fig_obv, use_container_width=True)
    
    st.markdown("""
    ### 🕵️ Was verrät das Volumen?
    Das **On-Balance Volume (OBV)** addiert das Volumen an positiven Tagen und subtrahiert es an negativen Tagen.
    
    * **Preis steigt + OBV steigt:** Der Trend ist gesund und durch echtes Geld gedeckt.
    * **Preis steigt + OBV fällt (Divergenz):** Warnsignal! Der Anstieg wird nicht durch Volumen gestützt (mögliche Trendwende).
    """)

# TAB 4: KI & News
with tab4:
    st.subheader("🤖 KI Vorhersage & News")
    
    # News holen
    with st.spinner('Analysiere News...'):
        news_df = get_news_and_sentiment(ticker)
        avg_sentiment = news_df['Sentiment_Score'].mean() if not news_df.empty else 0
    
    # Modell trainieren
    with st.spinner('Trainiere KI mit neuen Indikatoren...'):
        predictor = train_model(df)
        prediction = predictor.predict_with_sentiment(df, sentiment_score=avg_sentiment)

    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric("KI Prognose (Return)", f"{prediction['final_predicted_return']*100:.2f}%")
        st.metric("Erwarteter Preis", f"${prediction['predicted_price']:.2f}")
        
        st.write("---")
        st.write(f"News Stimmung: **{avg_sentiment:.2f}**")
        st.progress((avg_sentiment + 1) / 2)

    with col_res2:
        st.markdown("### Warum dieses Ergebnis?")
        st.write("Die KI hat folgende Faktoren gewichtet:")
        # Feature Importance auslesen
        importances = pd.DataFrame({
            'Feature': predictor.features,
            'Wichtigkeit': predictor.model.feature_importances_
        }).sort_values(by='Wichtigkeit', ascending=False).head(5)
        
        st.dataframe(importances, hide_index=True)

    if not news_df.empty:
        st.subheader("Letzte Schlagzeilen")
        st.dataframe(news_df[['Date', 'Title', 'Sentiment_Score']], hide_index=True)