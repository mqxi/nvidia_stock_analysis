# 🚀 NVIDIA Stock Intelligence & Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Package Manager](https://img.shields.io/badge/Manager-uv-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Ein Data-Science-Projekt zur ganzheitlichen Analyse der NVIDIA-Aktie (NVDA). Dieses Tool kombiniert klassische Chart-Analyse mit Machine Learning und News-Sentiment-Tracking, um fundierte Einblicke in die Kursentwicklung zu geben.

---

## 🎯 Features

### 1. Erweiterte Statistische Analyse
* **Echtzeit-Daten:** Abruf historischer und aktueller Kursdaten via `yfinance`.
* **Technische Indikatoren (Deep Dive):**
    * **Trend:** SMA (20/50), MACD (Moving Average Convergence Divergence).
    * **Momentum:** RSI (Relative Strength Index).
    * **Volatilität:** Bollinger Bands, ATR (Average True Range).
    * **Volumen:** OBV (On-Balance Volume).

### 2. Event & News Crawler
* **Smart Scraping:** Durchsucht Google News RSS-Feeds nach aktuellen Schlagzeilen zu NVIDIA.
* **Filterung:** Extrahiert automatisch Datum, Headline und Quelle.

### 3. Machine Learning & KI
* **Sentiment Analyse (NLP):** Bewertung von News-Headlines mittels **VADER** (Valence Aware Dictionary and sEntiment Reasoner) auf einer Skala von -1 (negativ) bis +1 (positiv).
* **Hybride Kurs-Vorhersage:** * Nutzt einen **Random Forest Regressor** (Scikit-Learn) trainiert auf relativen Renditen.
    * Kombiniert technische Signale mit dem aktuellen News-Sentiment für eine angepasste Prognose.
    * Feature Importance Analyse (zeigt, welche Indikatoren entscheidend sind).

### 4. Interactive Dashboard
* Moderne Web-App basierend auf **Streamlit**.
* Interaktive **Plotly**-Charts mit Zoom-Funktion und Tab-Navigation.

---

## 🛠 Tech Stack

* **Sprache:** Python 3.12+
* **Package Manager:** `uv`
* **Datenquelle:** Yahoo Finance API (`yfinance`)
* **Datenverarbeitung:** Pandas, NumPy
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
├── .venv/                 # Virtuelle Umgebung (von uv verwaltet)
├── data/                  # Lokaler Cache für CSV-Dateien/Logs
├── notebooks/             # Jupyter Notebooks für Experimente
├── tests/                 # Unit Tests
│
├── src/                   # Quellcode Module
│   ├── data_loader.py     # API-Verbindung zu Yahoo Finance
│   ├── scraper.py         # Google News RSS Parser
│   ├── indicators.py      # Berechnung (RSI, MACD, ATR, OBV)
│   ├── sentiment.py       # NLTK VADER Analyse
│   └── predictor.py       # Random Forest ML Modell
│
├── app.py                 # Hauptanwendung (Streamlit Entry Point)
├── pyproject.toml         # Projekt-Konfiguration & Dependencies
├── uv.lock                # Lockfile für reproduzierbare Builds
└── README.md              # Dokumentation
```
