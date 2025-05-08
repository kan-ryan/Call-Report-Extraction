import io
import os
import glob
import time
import yaml # type: ignore
import requests
import pandas as pd

TRANSCRIPT_ROOT = "data/Transcripts/"
PRICE_DIR = "data/prices/"
CONFIG_PATH = "config/api_keys.yaml"


def load_api_key():
    with open(CONFIG_PATH, "r") as f:
        keys = yaml.safe_load(f)
    return keys.get("alpha_vantage")


def download_stock_data(ticker, start_date="2018-01-01", end_date="2024-12-31"):
    api_key = load_api_key()
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "csv"
    }

    print(f"Requesting Alpha Vantage data for {ticker}...")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data for {ticker}: {response.status_code}")
        return None

    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        print(f"⚠️ No data returned for {ticker}.")
        return None

    os.makedirs(PRICE_DIR, exist_ok=True)
    save_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df

def load_transcripts():
    transcripts = []
    for path in glob.glob(f"{TRANSCRIPT_ROOT}/*/*.txt"):
        try:
            parts = os.path.normpath(path).split(os.sep)
            ticker = parts[-2]
            filename = parts[-1]
            date = filename.split("-")[:3]
            date_str = "-".join(date)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            transcripts.append({
                "ticker": ticker,
                "date": date_str,
                "filename": filename,
                "text": text
            })
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.DataFrame(transcripts)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA"]
    for t in tickers:
        df = download_stock_data(t)
        time.sleep(15)  

    df_transcripts = load_transcripts()
    print(f"Loaded {len(df_transcripts)} transcripts")
    print(df_transcripts.head())
