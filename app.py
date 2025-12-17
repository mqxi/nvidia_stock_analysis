import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.agents import HedgeFund

# Importiere unsere eigenen Module
from src.data_loader import load_stock_data
from src.indicators import add_indicators
from src.predictor import StockPredictor
from src.scraper import NewsScraper
from src.sentiment import SentimentAnalyzer

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
    mixed_df = scraper.get_all_sources(ticker)
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_news(mixed_df)


@st.cache_resource
def train_model(df):
    predictor = StockPredictor()
    predictor.train(df)
    return predictor


# --- Hauptlogik ---
with st.spinner("Lade komplexe Finanzdaten..."):
    df = get_data(ticker, period)

if df is None:
    st.error("Daten konnten nicht geladen werden.")
    st.stop()

# Key Metrics oben
latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Kurs", f"${latest['Close']:.2f}", f"{latest['Daily_Return'] * 100:.2f}%")
col2.metric(
    "RSI (Momentum)",
    f"{latest['RSI']:.0f}",
    "Neutral" if 30 < latest["RSI"] < 70 else "Extrem",
)
col3.metric("ATR (Volatilität)", f"${latest['ATR']:.2f}", "Schwankungsbreite")
col4.metric(
    "MACD Signal",
    "KAUFEN" if latest["MACD"] > latest["MACD_Signal"] else "VERKAUFEN",
    delta_color="normal",
)

# --- Tabs für bessere Übersicht ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📊 Chart",
        "📉 Momentum",
        "🌊 Volumen",
        "🧠 KI Prognose",
        "☁️ NLP",
        "🔬 Math & Cycles",
        "🕵️ Agenten Rat",
    ]
)

# TAB 1: Hauptchart
with tab1:
    st.subheader("Preisentwicklung & Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Upper"],
            line=dict(color="gray", width=1, dash="dot"),
            name="Upper Band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Lower"],
            line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty",
            name="Lower Band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_50"],
            line=dict(color="orange", width=2),
            name="SMA 50",
        )
    )

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

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
    fig_macd.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue"), name="MACD"),
        row=1,
        col=1,
    )
    fig_macd.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD_Signal"], line=dict(color="orange"), name="Signal"
        ),
        row=1,
        col=1,
    )

    # Histogramm
    colors = ["green" if val >= 0 else "red" for val in df["MACD_Hist"]]
    fig_macd.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=colors, name="Histogramm"),
        row=2,
        col=1,
    )

    fig_macd.update_layout(height=500)
    st.plotly_chart(fig_macd, width="stretch")

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
    fig_obv.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Preis"), row=1, col=1)
    fig_obv.add_trace(
        go.Scatter(x=df.index, y=df["OBV"], line=dict(color="purple"), name="OBV"),
        row=2,
        col=1,
    )

    st.plotly_chart(fig_obv, width="stretch")

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
    with st.spinner("Analysiere News..."):
        news_df = get_news_and_sentiment(ticker)
        avg_sentiment = news_df["Sentiment_Score"].mean() if not news_df.empty else 0

    # Modell trainieren
    with st.spinner("Trainiere KI mit neuen Indikatoren..."):
        predictor = train_model(df)
        prediction = predictor.predict_with_sentiment(df, sentiment_score=avg_sentiment)

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(
            "KI Prognose (Return)", f"{prediction['final_predicted_return'] * 100:.2f}%"
        )
        st.metric("Erwarteter Preis", f"${prediction['predicted_price']:.2f}")

        st.write("---")
        st.write(f"News Stimmung: **{avg_sentiment:.2f}**")
        st.progress((avg_sentiment + 1) / 2)

    with col_res2:
        st.markdown("### Warum dieses Ergebnis?")
        st.write("Die KI hat folgende Faktoren gewichtet:")
        # Feature Importance auslesen
        importances = (
            pd.DataFrame(
                {
                    "Feature": predictor.features,
                    "Wichtigkeit": predictor.model.feature_importances_,
                }
            )
            .sort_values(by="Wichtigkeit", ascending=False)
            .head(5)
        )

        st.dataframe(importances, hide_index=True)

    if not news_df.empty:
        st.subheader("Letzte Schlagzeilen")
        st.dataframe(news_df[["Date", "Title", "Sentiment_Score"]], hide_index=True)

# TAB 5: Social Sentiment & Insider Talk
with tab5:
    st.subheader("📢 Social Sentiment & Insider Talk")
    st.markdown(
        "Was denken die Privatanleger auf **Stocktwits** und **Reddit** im Vergleich zu den Medien?"
    )

    if not news_df.empty:
        # Daten aufteilen
        social_df = news_df[news_df["Type"] == "Social"]
        mainstream_df = news_df[news_df["Type"] == "News"]

        # --- Metrics Row ---
        col_s1, col_s2, col_s3 = st.columns(3)

        social_sent = social_df["Sentiment_Score"].mean() if not social_df.empty else 0
        news_sent = (
            mainstream_df["Sentiment_Score"].mean() if not mainstream_df.empty else 0
        )

        col_s1.metric(
            "Stimmung: Social Media",
            f"{social_sent:.2f}",
            "Bullish" if social_sent > 0.05 else "Bearish",
            delta_color="normal",
        )
        col_s2.metric(
            "Stimmung: Nachrichten",
            f"{news_sent:.2f}",
            "Positiv" if news_sent > 0.05 else "Negativ",
        )
        col_s3.metric("Anzahl Posts (24h)", len(news_df))

        st.markdown("---")

        # --- Layout: Links Feed, Rechts Analyse ---
        col_feed, col_viz = st.columns([0.4, 0.6])

        with col_feed:
            st.markdown("### 🔥 Live Feed (Stocktwits & Reddit)")
            # Eigener kleiner Feed-Reader
            for index, row in social_df.head(10).iterrows():
                with st.container():
                    # Icon je nach Quelle
                    icon = "🐦" if "Stocktwits" in row["Source"] else "👽"

                    # Sentiment Farbe
                    color = (
                        "green"
                        if row["Sentiment_Score"] > 0.2
                        else "red"
                        if row["Sentiment_Score"] < -0.2
                        else "gray"
                    )

                    st.markdown(
                        f"**{icon} {row['Source']}** <span style='color:{color}'>({row['Sentiment_Score']:.2f})</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{row['Date'].strftime('%H:%M')} | {row['Title']}")
                    st.divider()

        with col_viz:
            st.markdown("### ☁️ Worüber reden die Trader?")
            # Wordcloud nur aus Social Media Daten
            import matplotlib.pyplot as plt
            from wordcloud import WordCloud

            analyzer = SentimentAnalyzer()

            clean_text = analyzer.get_text_for_wordcloud(
                social_df if not social_df.empty else news_df
            )
            if len(clean_text) > 0:
                # Eigene Farben für Wordcloud (Orange/Weiß für Reddit/Social Style)
                wordcloud = WordCloud(
                    width=800, height=500, background_color="#0e1117", colormap="Wistia"
                ).generate(clean_text)
                fig_wc, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                # Hintergrund transparent machen für Dark Mode
                fig_wc.patch.set_facecolor("#0e1117")
                st.pyplot(fig_wc)
            else:
                st.info("Nicht genug Social Daten für eine Wolke.")

            st.markdown("### ⚖️ Stimmung vs. Subjektivität (Alle Quellen)")
            # Scatter Plot Code von vorhin (bleibt gleich, ist aber jetzt spannender)
            fig_scatter = go.Figure()
            # Social in Blau, News in Orange
            colors = news_df["Type"].map({"Social": "cyan", "News": "orange"})

            fig_scatter.add_trace(
                go.Scatter(
                    x=news_df["Sentiment_Score"],
                    y=news_df["Subjectivity"],
                    mode="markers",
                    text=news_df["Title"],
                    marker=dict(size=10, color=colors),
                )
            )
            fig_scatter.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Subjektivität",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_scatter, width="stretch")
            st.caption("🔵 = Social Media (Meinung) | 🟠 = News (Redaktionell)")

    else:
        st.warning("Keine Daten gefunden. API Limit oder Internet-Problem?")

# TAB 6: Mathematische Zeitreihen-Analyse
with tab6:
    st.subheader("Mathematische Zeitreihen-Analyse")
    st.markdown(
        "Identifikation von versteckten Mustern und Zyklen, die dem bloßen Auge verborgen bleiben."
    )

    # Berechnungen durchführen
    with st.spinner("Berechne Fourier-Transformation & Zerlegung..."):
        # Für Decomposition brauchen wir genug Daten (mind. 2 Jahre empfohlen für period=252)
        decomposition = None
        if len(df) > 300:
            from src.indicators import (
                calculate_fourier_transform,
                calculate_seasonal_decomposition,
            )

            decomposition = calculate_seasonal_decomposition(
                df, period=60
            )  # Quartals-Saison (ca. 60 Handelstage)
            fourier_df = calculate_fourier_transform(df)
        else:
            st.warning(
                "Für diese Analyse werden mindestens 2 Jahre Daten benötigt. Bitte Zeitraum in der Sidebar erhöhen."
            )

    if decomposition:
        # 1. Seasonal Decomposition Plot
        st.markdown("### 🧩 Time Series Decomposition (Trend vs. Saison)")
        st.caption(
            "Zerlegt den Kurs in drei Komponenten: Den langfristigen Trend, das wiederkehrende Muster (Saison) und das Rauschen (Noise)."
        )

        fig_decomp = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Langfristiger Trend",
                "Wiederkehrendes Muster (Saisonalität)",
                "Rauschen (Residuals)",
            ),
            vertical_spacing=0.1,
        )

        # Trend
        fig_decomp.add_trace(
            go.Scatter(
                x=df.index,
                y=decomposition["trend"],
                line=dict(color="blue"),
                name="Trend",
            ),
            row=1,
            col=1,
        )
        # Seasonal
        fig_decomp.add_trace(
            go.Scatter(
                x=df.index,
                y=decomposition["seasonal"],
                line=dict(color="green"),
                name="Saisonalität",
            ),
            row=2,
            col=1,
        )
        # Residuals
        fig_decomp.add_trace(
            go.Scatter(
                x=df.index,
                y=decomposition["resid"],
                mode="markers",
                marker=dict(color="gray", size=2),
                name="Residuals",
            ),
            row=3,
            col=1,
        )

        fig_decomp.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig_decomp, width="stretch")

        st.markdown("---")

        # 2. Fourier Transform Plot
        st.markdown("### 🌊 Fourier Analyse: Dominante Zyklen")
        st.caption(
            "Die Fast Fourier Transform (FFT) zeigt, welche Zyklen (in Tagen) den stärksten Einfluss auf den Kurs haben."
        )

        # Wir filtern Rauschen raus und zeigen nur Zyklen zwischen 10 und 365 Tagen
        fourier_filtered = fourier_df[
            (fourier_df["Cycle_Length_Days"] > 20)
            & (fourier_df["Cycle_Length_Days"] < 365)
        ].head(20)

        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Bar(
                x=fourier_filtered["Cycle_Length_Days"],
                y=fourier_filtered["Amplitude"],
                marker_color="purple",
            )
        )

        fig_fft.update_layout(
            xaxis_title="Zyklus-Länge (Tage)",
            yaxis_title="Stärke (Amplitude)",
            xaxis_type="log",  # Logarithmisch ist oft besser bei Frequenzen
            height=400,
            hovermode="x",
        )
        st.plotly_chart(fig_fft, width="stretch")

        # Top Zyklen Text
        top_cycle = fourier_filtered.iloc[0]["Cycle_Length_Days"]
        st.info(
            f"💡 **Insight:** Der stärkste erkannte Zyklus wiederholt sich etwa alle **{top_cycle:.1f} Tage**. Achte auf Muster in diesem Abstand!"
        )

# TAB 7: AI Agent Council
with tab7:
    st.subheader("🕵️ Der KI-Investoren Rat")
    st.markdown(
        "Wir simulieren ein Team aus drei Experten-Agenten, die unterschiedliche Daten analysieren und zu einem gemeinsamen Entschluss kommen."
    )

    # Wir brauchen Daten aus den anderen Modulen, die wir oben schon geladen haben
    # (df, news_df, prediction, decomposition) sind schon da

    fund = HedgeFund()

    # Decomposition muss eventuell neu berechnet werden, falls Tab 6 nicht geklickt wurde
    if "decomposition" not in locals() or decomposition is None:
        from src.indicators import calculate_seasonal_decomposition

        decomposition = calculate_seasonal_decomposition(df, period=60)

    # Analyse starten
    agents, verdict, color = fund.get_verdict(df, news_df, prediction, decomposition)

    # Großes Ergebnis anzeigen
    st.markdown("---")
    st.markdown(
        f"<h2 style='text-align: center; color: {color};'>Gesamturteil: {verdict}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Karten für jeden Agenten anzeigen
    col_a1, col_a2, col_a3 = st.columns(3)

    for col, agent in zip([col_a1, col_a2, col_a3], agents):
        with col:
            # Farbe für den Agenten Header
            header_color = (
                "green"
                if agent.vote == "BULLISH"
                else "red"
                if agent.vote == "BEARISH"
                else "gray"
            )
            st.markdown(f"### :{header_color}[{agent.name}]")
            st.caption(f"Rolle: {agent.role}")

            st.metric("Votum", agent.vote, f"Sicherheit: {agent.confidence * 100:.0f}%")

            with st.container(border=True):
                st.markdown("**Begründung:**")
                st.write(agent.reason)

    st.info(
        "💡 **Das Prinzip:** Multi-Agenten-Systeme reduzieren Fehler, indem sie nicht nur einer Datenquelle vertrauen (z.B. nur dem Chart), sondern technische, fundamentale und statistische Signale gegeneinander abwägen."
    )
