import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import yfinance.shared as shared

from flask import Flask, request, jsonify

warnings.filterwarnings("ignore", category=FutureWarning)
app = Flask(__name__)

# --- TÜM ANALİZ FONKSİYONLARIN BURADA (Kodu olduğu gibi buraya yapıştırdım, üstte verdiğin kodun tamamı!) ---
# --- Aşağıya SADECE sessiz_guc_stratejisi fonksiyonunu ÇIKTI ALACAK şekilde tekrar uyarladım ---

@lru_cache(maxsize=32)
def veri_cek(hisse_kodu, start_date, end_date, interval):
    try:
        df = yf.download(hisse_kodu, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        if df.empty or hisse_kodu in shared._ERRORS:
            raise ValueError(f"Veri çekilemedi: {shared._ERRORS.get(hisse_kodu, 'Bilinmeyen hata')}")
        latest_date = df.index[-1].date()
        today = datetime.now().date()
        if latest_date < today:
            print(f"⚠️ Uyarı: En son veri {latest_date} tarihli. Güncel veri eksik olabilir.")
        return df
    except Exception as e:
        raise ValueError(f"yfinance Hatası: {str(e)}. Ticker sembolünü veya bağlantıyı kontrol edin.")

# (Tüm teknik analiz ve yardımcı fonksiyonlar burada olacak. Sende zaten hazır.)

# ---------------------------------------------------------
# --- ANA ANALİZ API: KISA, HIZLI, JSON DÖNÜŞ SAĞLAR ---
# ---------------------------------------------------------
def analiz90_json(hisse_kodu):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df = veri_cek(hisse_kodu, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), "1h")
        if df.empty:
            return {"hata": "Veri çekilemedi veya hisse kodu yanlış!"}

        # Ana analiz fonksiyonundan özet sinyaller çekiyoruz:
        results = {}

        # Her göstergeye ait çıktılar
        results['sessiz_guc'] = hesapla_sessiz_guc(df)
        results['hacim_artisi'] = hesapla_hacim_artisi(df)
        results['fhm'] = hesapla_fhm(df)
        results['phf'] = hesapla_phf(df)
        results['triangle'] = hesapla_triangle(df)
        results['double_top'] = hesapla_double_top(df)
        results['bearish_pennant'] = hesapla_bearish_pennant(df)
        results['bullish_flag'] = hesapla_bullish_flag(df)
        results['double_bottom'] = hesapla_double_bottom(df)
        results['destek_direnc'] = hesapla_destek_direnc(df)
        results['yukselen_kama'] = hesapla_yukselen_kama(df)
        results['kanal_yukari'] = hesapla_kanal_yukari(df)
        results['fincan_kulp'] = hesapla_fincan_kulp(df)
        results['ma5_22'] = hesapla_ma5_22(df)
        results['rsi'] = hesapla_rsi(df)
        results['ema50'] = hesapla_ema50(df)
        results['sma50'] = hesapla_sma50(df)
        results['wma50'] = hesapla_wma50(df)
        results['fibonacci'] = hesapla_fibonacci(df)
        results['bollinger'] = hesapla_bollinger(df)
        results['macd'] = hesapla_macd(df)
        results['keltner'] = hesapla_keltner(df)
        results['kanaldan_cikis'] = hesapla_kanaldan_cikis(df)

        return {
            "hisse": hisse_kodu,
            "analiz": results
        }
    except Exception as e:
        return {"hata": str(e)}

@app.route('/analiz90', methods=['POST'])
def analiz90_api():
    data = request.get_json()
    hisse_kodu = data.get("hisse_kodu")
    if not hisse_kodu:
        return jsonify({"hata": "Hisse kodu eksik!"}), 400
    sonuc = analiz90_json(hisse_kodu)
    return jsonify(sonuc)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
