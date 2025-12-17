# 🚀 NVIDIA Stock Intelligence & Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Package Manager](https://img.shields.io/badge/Manager-uv-purple)
![AI Agents](https://img.shields.io/badge/AI-Multi--Agent_System-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Ein Data-Science-Projekt zur ganzheitlichen Analyse der NVIDIA-Aktie (NVDA). Dieses Tool kombiniert klassische Chart-Analyse mit Machine Learning und News-Sentiment-Tracking, um fundierte Einblicke in die Kursentwicklung zu geben.

---

## 🎯 Features

### 1. 📊 Marktdaten & Chart-Analyse
* **Live-Daten:** Abruf aktueller Kurse und Historie via Yahoo Finance API.
* **Interaktive Charts:** Zoom-bare Candlestick-Charts (Plotly) für detaillierte Einblicke.
* **Technische Indikatoren:** Automatische Berechnung der wichtigsten Metriken für Trader:
    * **Trend:** MACD & SMA (20/50 Tage).
    * **Momentum:** RSI (Relative Strength Index).
    * **Volatilität:** Bollinger Bands & ATR.
    * **Volumen:** OBV (On-Balance Volume) zur Erkennung von "Smart Money" Flüssen.

### 2. 📢 News & Social Sentiment (Die Stimmung)
* **Stealth Scraper:** Crawlt Daten von **Google News**, **Stocktwits** und **Reddit** (r/nvidia, r/wallstreetbets) und umgeht dabei Bot-Schutzmechanismen.
* **NLP Deep Dive:**
    * **Stimmung:** Bewertet Headlines als Positiv/Negativ (VADER).
    * **Subjektivität:** Unterscheidet zwischen harten Fakten und bloßen Meinungen (TextBlob).
    * **WordCloud:** Visualisiert, worüber der Markt gerade spricht (z.B. "AI Chips", "China", "Earnings").

### 3. 🧠 Machine Learning & Mathematik (Der Quant-Ansatz)
Einsatz von Algorithmen, zur Mustererkennung.
* **KI-Prognose:** Ein **Random Forest Regressor** lernt aus historischen Mustern, um die relative Rendite (Return) für den nächsten Tag vorherzusagen.
* **Feature Importance:** Zeigt transparent an, welche Indikatoren (z.B. Volumen vs. RSI) die KI-Entscheidung gerade treiben.
* **Zyklus-Analyse:**
    * **Fourier-Transformation:** Deckt versteckte, wiederkehrende Zeit-Zyklen auf (z.B. "Alle 90 Tage ein Hoch").
    * **Seasonal Decomposition:** Zerlegt den Kurs in langfristigen Trend, Saisonalität und Rauschen.

### 4. 🕵️ Multi-Agent System (Das Highlight)
Ein simuliertes "Hedge-Fonds-Komitee", das alle oben genannten Daten zusammenführt und diskutiert.
* **Dr. Chart (Technical Agent):** Entscheidet rein nach Chart-Signalen.
* **Mr. Hype (Sentiment Agent):** Achtet nur auf die Stimmung der Privatanleger.
* **The Brain (Quant Agent):** Vertraut nur der KI und der Mathematik.
* **Konsens-Findung:** Am Ende geben die Agenten ein gemeinsames Votum ab (**KAUFEN, HALTEN** oder **VERKAUFEN**) inkl. Begründung.

---

## 🛠 Tech Stack

* **Sprache:** Python 3.12+
* **Package Manager:** `uv`
* **Datenquelle:** Yahoo Finance API (`yfinance`)
* **Datenverarbeitung:** Pandas, NumPy, Scikit-Learn, Statsmodels, SciPy
* **Visualisierung:** Plotly, Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest), NLTK (VADER)
* **Web Scraping:** BeautifulSoup4, Requests, lxml

## Setup

```bash
uv sync              # Install dependencies
uv add <package>     # Add new package
uv run streamlit run app.py
```

## 📂 Projektstruktur

```
nvidia_stock_analysis/
│
├── src/                   # Core Logic
│   ├── agents.py          # Die KI-Agenten (Dr. Chart, Mr. Hype, The Brain)
│   ├── data_loader.py     # yfinance API Wrapper
│   ├── indicators.py      # Mathematik (RSI, MACD, Fourier, Decomposition)
│   ├── predictor.py       # Random Forest ML Modell
│   ├── scraper.py         # Google/Stocktwits/Reddit Scraper (Stealth Mode)
│   └── sentiment.py       # NLP Logik (VADER, TextBlob, WordCloud)
│
├── app.py                 # Hauptanwendung (Streamlit Entry Point)
├── pyproject.toml         # Projekt-Konfiguration & Dependencies
└── README.md              # Dokumentation
```
