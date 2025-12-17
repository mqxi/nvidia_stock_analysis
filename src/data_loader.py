import os

import yfinance as yf


def load_stock_data(ticker_symbol, period="5y", interval="1d"):
    """
    Lädt historische Aktiendaten von Yahoo Finance.

    Args:
        ticker_symbol (str): Das Symbol der Aktie (z.B. "NVDA").
        period (str): Zeitraum der Daten (z.B. "1y", "5y", "max").
        interval (str): Datenintervall (z.B. "1d" für täglich, "1wk" für wöchentlich).

    Returns:
        pd.DataFrame: DataFrame mit den Spalten Open, High, Low, Close, Volume.
    """
    print(f"🔄 Lade Daten für {ticker_symbol} ({period})...")

    # Ticker-Objekt initialisieren
    ticker = yf.Ticker(ticker_symbol)

    # Historische Daten abrufen
    # auto_adjust=True korrigiert Splits und Dividenden im Close-Preis
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print(f"⚠️ Warnung: Keine Daten für {ticker_symbol} gefunden.")
        return None

    # Zeitzone entfernen, falls vorhanden (macht Plotting einfacher)
    df.index = df.index.tz_localize(None)

    print(f"✅ {len(df)} Datensätze geladen.")
    return df


def save_to_csv(df, filename):
    """Speichert den DataFrame als CSV im data-Ordner."""
    directory = "data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, filename)
    df.to_csv(path)
    print(f"💾 Daten gespeichert unter: {path}")


# --- Test-Bereich ---
if __name__ == "__main__":
    # Testlauf
    symbol = "NVDA"
    data = load_stock_data(symbol)

    if data is not None:
        print(data.head())  # Zeige die ersten 5 Zeilen
        save_to_csv(data, f"{symbol}_history.csv")
