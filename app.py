import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
app = Flask(__name__)

# -----------------------------------
# Teknik Gösterge Fonksiyonları
# -----------------------------------

def hesapla_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] > 70:
        return "🔺 RSI: AŞIRI ALIM"
    elif rsi.iloc[-1] < 30:
        return "🔻 RSI: AŞIRI SATIM"
    else:
        return "📊 RSI: NÖTR"

def hesapla_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1]:
        return "🔺 MACD: AL"
    elif macd.iloc[-1] < signal.iloc[-1]:
        return "🔻 MACD: SAT"
    else:
        return "📊 MACD: NÖTR"

def hesapla_ema50(df):
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    if df['Close'].iloc[-1] > ema50.iloc[-1]:
        return "📈 EMA50: ÜZERİNDE (POZİTİF)"
    else:
        return "📉 EMA50: ALTINDA (NEGATİF)"

# -----------------------------------
# Veri Çekme Fonksiyonu
# -----------------------------------

def veri_cek(hisse_kodu):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    df = yf.download(hisse_kodu, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1h")
    return df

# -----------------------------------
# Flask API Endpoint
# -----------------------------------

@app.route('/analiz90', methods=['POST'])
def analiz90_api():
    data = request.get_json()
    hisse_kodu = data.get("hisse_kodu")
    if not hisse_kodu:
        return jsonify({"hata": "Hisse kodu eksik!"}), 400
    try:
        df = veri_cek(hisse_kodu)
        if df.empty:
            return jsonify({"hata": "Veri çekilemedi."}), 404

        sonuc = {
            "RSI": hesapla_rsi(df),
            "MACD": hesapla_macd(df),
            "EMA50": hesapla_ema50(df),
        }
        return jsonify({
            "hisse": hisse_kodu,
            "analiz": sonuc
        })
    except Exception as e:
        return jsonify({"hata": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
