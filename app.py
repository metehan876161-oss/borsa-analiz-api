import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import yfinance.shared as shared

# Gelecek uyarılarını bastır
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Veri çekme fonksiyonu (cache ile)
@lru_cache(maxsize=32)
def veri_cek(hisse_kodu, start_date, end_date, interval):
    try:
        df = yf.download(hisse_kodu, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        if df.empty or hisse_kodu in shared._ERRORS:
            raise ValueError(f"Veri çekilemedi: {shared._ERRORS.get(hisse_kodu, 'Bilinmeyen hata')}")
        latest_date = df.index[-1].date()
        today = datetime.now().date()
        if latest_date < today:
            return df, f"⚠️ Uyarı: En son veri {latest_date} tarihli. Güncel veri eksik olabilir."
        return df, None
    except Exception as e:
        raise ValueError(f"yfinance Hatası: {str(e)}. Ticker sembolünü veya bağlantıyı kontrol edin.")

# Sessiz Güç Stratejisi
def hesapla_sessiz_guc(df, hacim_katsayisi=1.1):
    volume_last_10 = df['Volume'].tail(10).mean()
    volume_prev_10 = df['Volume'].tail(20).head(10).mean()
    hacim_azaliyor = volume_last_10 < volume_prev_10

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    today_close = float(today['Close'])
    today_open = float(today['Open'])
    yesterday_open = float(yesterday['Open'])
    yesterday_close = float(yesterday['Close'])
    yutan_boga = (today_close > yesterday_open) and (today_open < yesterday_close) and (today_close > yesterday_close)
    today_body = abs(today_close - today_open)
    yesterday_body = abs(yesterday_close - yesterday_open)
    yutan_boga = yutan_boga and (today_body > yesterday_body * hacim_katsayisi)

    hacim_ortalama = df['Volume'].tail(6).iloc[:-1].mean()
    bugun_hacim = float(today['Volume'])
    hacim_artisi = bugun_hacim > (hacim_ortalama * hacim_katsayisi) and (today_close > yesterday_close)

    return {
        'Koşul': ['Hacim Azalma Trendi', 'Yutan Boğa Formasyonu', 'Hacim Patlaması'],
        'Durum': ['Evet' if hacim_azaliyor else 'Hayır', 'Evet' if yutan_boga else 'Hayır', 'Evet' if hacim_artisi else 'Hayır'],
        'Sinyal': ['Alım' if hacim_azaliyor or yutan_boga or hacim_artisi else 'Yok', 'Alım' if yutan_boga else 'Yok', 'Alım' if hacim_artisi else 'Yok'],
        'Detay': [
            f"Son 10 gün: {float(volume_last_10):.2f}, Önceki 10 gün: {float(volume_prev_10):.2f}",
            f"Bugün gövde: {float(today_body):.2f}, Dün gövde: {float(yesterday_body):.2f}",
            f"Bugün hacim: {float(bugun_hacim):.2f}, Ortalama: {float(hacim_ortalama):.2f}"
        ]
    }

# Hacim Artışı
def hesapla_hacim_artisi(df, hacim_katsayisi=1.1):
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    bugun_hacim = float(today['Volume'])
    dun_hacim = float(yesterday['Volume'])
    today_close = float(today['Close'])
    yesterday_close = float(yesterday['Close'])
    hacim_artisi = bugun_hacim > (dun_hacim * hacim_katsayisi) and (today_close > yesterday_close)
    fiyat_degisim_yuzde = ((today_close - yesterday_close) / yesterday_close) * 100
    uyari = ""
    if hacim_artisi and fiyat_degisim_yuzde < -5:
        uyari = f"⚠️ Uyarı: Hacim artışı var ancak fiyat %{fiyat_degisim_yuzde:.2f} düşüş gösterdi!"

    return {
        'Koşul': ['Bir Önceki Güne Göre Hacim Artışı'],
        'Durum': ['Evet' if hacim_artisi else 'Hayır'],
        'Sinyal': ['Alım' if hacim_artisi else 'Yok'],
        'Detay': [f"Bugün hacim: {bugun_hacim:.2f}, Dün hacim: {dun_hacim:.2f}, Fiyat değişimi: {fiyat_degisim_yuzde:.2f}%"],
        'Uyarı': [uyari]
    }

# FHM Döngüsü
def hesapla_fhm(df, hacim_katsayisi=1.1):
    df['Price_Change'] = df['Close'].pct_change() * 100
    last_3_price_change = df['Price_Change'].tail(3).mean()
    prev_3_price_change = df['Price_Change'].tail(6).head(3).mean()
    fiyat_momentum = last_3_price_change > prev_3_price_change * 1.2 and last_3_price_change > 0

    last_3_volumes = df['Volume'].tail(3)
    hacim_dongusu = (last_3_volumes.iloc[0] > last_3_volumes.iloc[1]) and (last_3_volumes.iloc[2] > last_3_volumes.iloc[1] * hacim_katsayisi) and (df['Close'].iloc[-1] > df['Close'].iloc[-2])

    last_5_data = df[['Price_Change', 'Volume']].tail(5)
    fiyat_hacim_korelasyon = last_5_data['Price_Change'].corr(last_5_data['Volume']) > 0.3

    df['Volatility'] = (df['High'] - df['Low']) / df['Close']
    avg_volatility = df['Volatility'].tail(10).mean()
    today_volatility = df['Volatility'].iloc[-1]
    volatilite_siniri = today_volatility < avg_volatility

    return {
        'Koşul': ['Fiyat Momentum Artışı', 'Hacim Döngüsü', 'Fiyat-Hacim Korelasyonu', 'Volatilite Sınırı'],
        'Durum': ['Evet' if fiyat_momentum else 'Hayır', 'Evet' if hacim_dongusu else 'Hayır', 'Evet' if fiyat_hacim_korelasyon else 'Hayır', 'Evet' if volatilite_siniri else 'Hayır'],
        'Sinyal': ['Alım' if fiyat_momentum or hacim_dongusu or fiyat_hacim_korelasyon or volatilite_siniri else 'Yok', 'Alım' if hacim_dongusu else 'Yok', 'Alım' if fiyat_hacim_korelasyon else 'Yok', 'Alım' if volatilite_siniri else 'Yok'],
        'Detay': [
            f"Son 3 gün: {last_3_price_change:.2f}%, Önceki 3 gün: {prev_3_price_change:.2f}%",
            f"Hacim trendi: {list(last_3_volumes.values)}",
            f"Korelasyon: {last_5_data['Price_Change'].corr(last_5_data['Volume']):.2f}",
            f"Bugün volatilite: {today_volatility:.2f}, Ortalama: {avg_volatility:.2f}"
        ]
    }

# Patlayıcı Hız Formasyonu (PHF)
def hesapla_phf(df, hacim_katsayisi=1.1):
    last_5_closes = df['Close'].tail(5)
    avg_price_change = df['Price_Change'].tail(20).mean()
    fiyat_hizlanmasi = all(last_5_closes.pct_change().dropna() > (avg_price_change / 100)) and (last_5_closes.iloc[-1] > last_5_closes.iloc[-2])

    avg_volume_10 = df['Volume'].tail(10).mean()
    bugun_hacim = float(df['Volume'].iloc[-1])
    hacim_kirilmasi = bugun_hacim > avg_volume_10 * hacim_katsayisi and (df['Close'].iloc[-1] > df['Close'].iloc[-2])

    last_5_change = df['Price_Change'].tail(5).mean()
    prev_20_change = df['Price_Change'].tail(20).mean()
    goreli_fiyat_gucu = last_5_change > prev_20_change * hacim_katsayisi and last_5_change > 0

    last_5_volatility = df['Volatility'].tail(5).mean()
    prev_5_volatility = df['Volatility'].tail(10).head(5).mean()
    daralan_volatilite = last_5_volatility < prev_5_volatility

    return {
        'Koşul': ['Fiyat Hızlanması', 'Hacim Kırılması', 'Göreli Fiyat Gücü', 'Daralan Volatilite'],
        'Durum': ['Evet' if fiyat_hizlanmasi else 'Hayır', 'Evet' if hacim_kirilmasi else 'Hayır', 'Evet' if goreli_fiyat_gucu else 'Hayır', 'Evet' if daralan_volatilite else 'Hayır'],
        'Sinyal': ['Alım' if fiyat_hizlanmasi or hacim_kirilmasi or goreli_fiyat_gucu or daralan_volatilite else 'Yok', 'Alım' if hacim_kirilmasi else 'Yok', 'Alım' if goreli_fiyat_gucu else 'Yok', 'Alım' if daralan_volatilite else 'Yok'],
        'Detay': [
            f"Son 5 gün artış: {list(last_5_closes.pct_change().dropna().values * 100)}%",
            f"Bugün hacim: {bugun_hacim:.2f}, 10 gün ort.: {avg_volume_10:.2f}",
            f"Son 5 gün: {last_5_change:.2f}%, 20 gün ort.: {prev_20_change:.2f}%",
            f"Son 5 gün volatilite: {last_5_volatility:.2f}, Önceki 5 gün: {prev_5_volatility:.2f}"
        ]
    }

# Üçgen Formasyonu
def hesapla_triangle(df, window=30, tolerans=0.03):
    highs = df['High'].tail(window)
    lows = df['Low'].tail(window)

    x = np.arange(len(highs))
    high_slope, high_intercept = np.polyfit(x, highs, 1)
    low_slope, low_intercept = np.polyfit(x, lows, 1)

    converging = abs(high_slope) > 0 and abs(low_slope) > 0 and high_slope < 0 and low_slope > 0
    current_price = df['Close'].iloc[-1]
    upper_trendline = high_intercept + high_slope * (len(highs) - 1)
    lower_trendline = low_intercept + low_slope * (len(lows) - 1)
    breakout_up = current_price > upper_trendline and df['Close'].iloc[-2] <= upper_trendline
    breakout_down = current_price < lower_trendline and df['Close'].iloc[-2] >= lower_trendline

    volume_trend = df['Volume'].tail(10).mean() < df['Volume'].tail(window).mean()

    triangle_sinyal = 'Alım' if converging and breakout_up and volume_trend else 'Satım' if converging and breakout_down and volume_trend else 'Yok'

    return {
        'Koşul': ['Üçgen Formasyonu'],
        'Durum': ['Evet' if converging and (breakout_up or breakout_down) else 'Hayır'],
        'Sinyal': [triangle_sinyal],
        'Detay': [f"Üst trend: {upper_trendline:.2f}, Alt trend: {lower_trendline:.2f}, Bugün: {current_price:.2f}"]
    }

# Çift Tepe (Double Top)
def hesapla_double_top(df, window=30, tolerans=0.03):
    last_30_highs = df['High'].tail(window)
    peaks = last_30_highs.nlargest(2)
    double_top = abs(peaks.iloc[0] - peaks.iloc[1]) < (last_30_highs.mean() * tolerans)
    neckline = df['Low'].tail(window).nsmallest(2).mean()
    breakout = double_top and (df['Close'].iloc[-1] < neckline)

    return {
        'Koşul': ['Çift Tepe (Double Top)'],
        'Durum': ['Evet' if breakout else 'Hayır'],
        'Sinyal': ['Satım' if breakout else 'Yok'],
        'Detay': [f"Tepe: {peaks.values}, Neckline: {neckline:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Flama Ayı (Bearish Pennant)
def hesapla_bearish_pennant(df, window=20, tolerans=0.03):
    price_drop = df['Close'].tail(window).pct_change().mean() < -0.01
    last_10_highs = df['High'].tail(10)
    last_10_lows = df['Low'].tail(10)
    x = np.arange(10)
    high_slope, _ = np.polyfit(x, last_10_highs, 1)
    low_slope, _ = np.polyfit(x, last_10_lows, 1)
    converging = high_slope < 0 and low_slope > 0
    current_price = df['Close'].iloc[-1]
    lower_trendline = last_10_lows.min()
    breakout_down = converging and (current_price < lower_trendline) and (df['Close'].iloc[-2] >= lower_trendline)

    return {
        'Koşul': ['Flama Ayı (Bearish Pennant)'],
        'Durum': ['Evet' if price_drop and converging and breakout_down else 'Hayır'],
        'Sinyal': ['Satım' if price_drop and converging and breakout_down else 'Yok'],
        'Detay': [f"Fiyat düşüşü: {price_drop}, Alt trend: {lower_trendline:.2f}, Bugün: {current_price:.2f}"]
    }

# Boğa Bayrağı (Bullish Flag)
def hesapla_bullish_flag(df, window=20, tolerans=0.03):
    price_rise = df['Close'].tail(window).pct_change().mean() > 0.01
    last_10_highs = df['High'].tail(10)
    last_10_lows = df['Low'].tail(10)
    x = np.arange(10)
    high_slope, _ = np.polyfit(x, last_10_highs, 1)
    low_slope, _ = np.polyfit(x, last_10_lows, 1)
    parallel = abs(high_slope - low_slope) < tolerans
    current_price = df['Close'].iloc[-1]
    upper_trendline = last_10_highs.max()
    breakout_up = parallel and (current_price > upper_trendline) and (df['Close'].iloc[-2] <= upper_trendline)

    return {
        'Koşul': ['Boğa Bayrağı (Bullish Flag)'],
        'Durum': ['Evet' if price_rise and parallel and breakout_up else 'Hayır'],
        'Sinyal': ['Alım' if price_rise and parallel and breakout_up else 'Yok'],
        'Detay': [f"Fiyat artışı: {price_rise}, Üst trend: {upper_trendline:.2f}, Bugün: {current_price:.2f}"]
    }

# Çift Dip (Double Bottom)
def hesapla_double_bottom(df, window=30, tolerans=0.03):
    last_30_lows = df['Low'].tail(window)
    dips = last_30_lows.nsmallest(2)
    double_bottom = abs(dips.iloc[0] - dips.iloc[1]) < (last_30_lows.mean() * tolerans)
    neckline = df['High'].tail(window).nlargest(2).mean()
    breakout = double_bottom and (df['Close'].iloc[-1] > neckline)

    return {
        'Koşul': ['Çift Dip (Double Bottom)'],
        'Durum': ['Evet' if breakout else 'Hayır'],
        'Sinyal': ['Alım' if breakout else 'Yok'],
        'Detay': [f"Dip: {dips.values}, Neckline: {neckline:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Destek ve Direnç Seviyeleri
def hesapla_destek_direnc(df, window=30, tolerans=0.03):
    last_30_highs = df['High'].tail(window)
    last_30_lows = df['Low'].tail(window)

    support = last_30_lows.round(2).value_counts().index[0]
    resistance = last_30_highs.round(2).value_counts().index[0]

    current_price = df['Close'].iloc[-1]
    support_proximity = abs(current_price - support) / support < tolerans
    resistance_proximity = abs(current_price - resistance) / resistance < tolerans
    support_breakout = current_price > support and df['Close'].iloc[-2] <= support
    resistance_breakout = current_price < resistance and df['Close'].iloc[-2] >= resistance

    sinyal = 'Alım' if support_proximity or support_breakout else 'Satım' if resistance_proximity or resistance_breakout else 'Yok'

    return {
        'Koşul': ['Destek-Direnç Seviyeleri'],
        'Durum': ['Evet' if support_proximity or resistance_proximity or support_breakout or resistance_breakout else 'Hayır'],
        'Sinyal': [sinyal],
        'Detay': [f"Destek: {support:.2f}, Direnç: {resistance:.2f}, Bugün: {current_price:.2f}"]
    }

# Yükselen Kama
def hesapla_yukselen_kama(df, window=30, tolerans=0.03):
    highs = df['High'].tail(window)
    lows = df['Low'].tail(window)

    x = np.arange(len(highs))
    high_slope, high_intercept = np.polyfit(x, highs, 1)
    low_slope, low_intercept = np.polyfit(x, lows, 1)

    converging = high_slope > 0 and low_slope > 0 and high_slope < low_slope
    current_price = df['Close'].iloc[-1]
    lower_trendline = low_intercept + low_slope * (len(lows) - 1)
    breakout_down = converging and (current_price < lower_trendline) and (df['Close'].iloc[-2] >= lower_trendline)

    return {
        'Koşul': ['Yükselen Kama (Rising Wedge)'],
        'Durum': ['Evet' if converging and breakout_down else 'Hayır'],
        'Sinyal': ['Satım' if converging and breakout_down else 'Yok'],
        'Detay': [f"Üst trend eğimi: {high_slope:.4f}, Alt trend eğimi: {low_slope:.4f}, Bugün: {current_price:.2f}"]
    }

# Yükselen Kanal
def hesapla_kanal_yukari(df, window=30, tolerans=0.03):
    highs = df['High'].tail(window)
    lows = df['Low'].tail(window)

    x = np.arange(len(highs))
    high_slope, high_intercept = np.polyfit(x, highs, 1)
    low_slope, low_intercept = np.polyfit(x, lows, 1)

    parallel = abs(high_slope - low_slope) < tolerans and high_slope > 0 and low_slope > 0
    current_price = df['Close'].iloc[-1]
    upper_trendline = high_intercept + high_slope * (len(highs) - 1)
    lower_trendline = low_intercept + low_slope * (len(lows) - 1)
    support_proximity = abs(current_price - lower_trendline) / lower_trendline < tolerans
    breakout_up = current_price > upper_trendline and df['Close'].iloc[-2] <= upper_trendline
    breakout_down = current_price < lower_trendline and df['Close'].iloc[-2] >= lower_trendline

    sinyal = 'Alım' if parallel and (support_proximity or breakout_up) else 'Satım' if parallel and breakout_down else 'Yok'

    return {
        'Koşul': ['Kanal Yukarı (Ascending Channel)'],
        'Durum': ['Evet' if parallel and (support_proximity or breakout_up or breakout_down) else 'Hayır'],
        'Sinyal': [sinyal],
        'Detay': [f"Üst trend: {upper_trendline:.2f}, Alt trend: {lower_trendline:.2f}, Bugün: {current_price:.2f}"]
    }

# Fincan Kulp
def hesapla_fincan_kulp(df):
    cup_base = df['Close'].tail(30).min()
    cup_peak = df['Close'].tail(10).max()
    handle = df['Close'].tail(5).pct_change().mean() > 0
    fincan_kulp = (cup_peak > cup_base * 1.2) and handle
    return {
        'Koşul': ['Fincan Kulp'],
        'Durum': ['Evet' if fincan_kulp else 'Hayır'],
        'Sinyal': ['Alım' if fincan_kulp else 'Yok'],
        'Detay': [f"Taban: {cup_base:.2f}, Pik: {cup_peak:.2f}, Handle trend: {df['Close'].tail(5).pct_change().mean():.2f}%"]
    }

# 5/22 Hareketli Ortalama
def hesapla_ma5_22(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA22'] = df['Close'].rolling(window=22).mean()
    ma5_22 = df['MA5'].iloc[-1] > df['MA22'].iloc[-1]
    ma5_22_sinyal = 'Alım' if ma5_22 else 'Satım' if df['MA5'].iloc[-1] < df['MA22'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['5/22 MA Kesişimi'],
        'Durum': ['Evet' if ma5_22 or df['MA5'].iloc[-1] < df['MA22'].iloc[-1] else 'Hayır'],
        'Sinyal': [ma5_22_sinyal],
        'Detay': [f"MA5: {df['MA5'].iloc[-1]:.2f}, MA22: {df['MA22'].iloc[-1]:.2f}"]
    }

# RSI
def hesapla_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_sinyal = 'Alım' if rsi.iloc[-1] < 30 else 'Satım' if rsi.iloc[-1] > 70 else 'Yok'
    return {
        'Koşul': ['RSI Durumu'],
        'Durum': ['Evet' if rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70 else 'Hayır'],
        'Sinyal': [rsi_sinyal],
        'Detay': [f"RSI: {rsi.iloc[-1]:.2f}"]
    }

# EMA50
def hesapla_ema50(df):
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    ema50_kirim = df['Close'].iloc[-1] > df['EMA50'].iloc[-1]
    ema50_sinyal = 'Alım' if ema50_kirim else 'Satım' if df['Close'].iloc[-1] < df['EMA50'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['EMA50 Kırılımı'],
        'Durum': ['Evet' if ema50_kirim or df['Close'].iloc[-1] < df['EMA50'].iloc[-1] else 'Hayır'],
        'Sinyal': [ema50_sinyal],
        'Detay': [f"EMA50: {df['EMA50'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# SMA50
def hesapla_sma50(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    sma50_kirim = df['Close'].iloc[-1] > df['SMA50'].iloc[-1]
    sma50_sinyal = 'Alım' if sma50_kirim else 'Satım' if df['Close'].iloc[-1] < df['SMA50'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['SMA50 Kırılımı'],
        'Durum': ['Evet' if sma50_kirim or df['Close'].iloc[-1] < df['SMA50'].iloc[-1] else 'Hayır'],
        'Sinyal': [sma50_sinyal],
        'Detay': [f"SMA50: {df['SMA50'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# WMA50
def hesapla_wma50(df):
    weights = np.arange(1, 51)
    df['WMA50'] = df['Close'].rolling(window=50).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    wma50_kirim = df['Close'].iloc[-1] > df['WMA50'].iloc[-1]
    wma50_sinyal = 'Alım' if wma50_kirim else 'Satım' if df['Close'].iloc[-1] < df['WMA50'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['WMA50 Kırılımı'],
        'Durum': ['Evet' if wma50_kirim or df['Close'].iloc[-1] < df['WMA50'].iloc[-1] else 'Hayır'],
        'Sinyal': [wma50_sinyal],
        'Detay': [f"WMA50: {df['WMA50'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Fibonacci
def hesapla_fibonacci(df):
    high_20 = df['High'].tail(20).max()
    low_20 = df['Low'].tail(20).min()
    fib_range = high_20 - low_20
    fib_382 = low_20 + (fib_range * 0.382)
    fib_50 = low_20 + (fib_range * 0.5)
    fib_sinyal = 'Alım' if (df['Close'].iloc[-1] > fib_382) and (df['Close'].iloc[-1] < fib_50) else 'Yok'
    return {
        'Koşul': ['Fibonacci %38.2-%50 Destek'],
        'Durum': ['Evet' if (df['Close'].iloc[-1] > fib_382) and (df['Close'].iloc[-1] < fib_50) else 'Hayır'],
        'Sinyal': [fib_sinyal],
        'Detay': [f"%38.2: {fib_382:.2f}, %50: {fib_50:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Bollinger Bantları
def hesapla_bollinger(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + (df['STD20'] * 2)
    df['LowerBB'] = df['MA20'] - (df['STD20'] * 2)
    bb_sinyal = 'Alım' if df['Close'].iloc[-1] < df['LowerBB'].iloc[-1] else 'Satım' if df['Close'].iloc[-1] > df['UpperBB'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['Bollinger Bant Kırılımı'],
        'Durum': ['Evet' if df['Close'].iloc[-1] < df['LowerBB'].iloc[-1] or df['Close'].iloc[-1] > df['UpperBB'].iloc[-1] else 'Hayır'],
        'Sinyal': [bb_sinyal],
        'Detay': [f"Alt: {df['LowerBB'].iloc[-1]:.2f}, Üst: {df['UpperBB'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# MACD
def hesapla_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_sinyal = 'Alım' if macd.iloc[-1] > signal.iloc[-1] else 'Satım' if macd.iloc[-1] < signal.iloc[-1] else 'Yok'
    return {
        'Koşul': ['MACD Kesişimi'],
        'Durum': ['Evet' if macd.iloc[-1] > signal.iloc[-1] or macd.iloc[-1] < signal.iloc[-1] else 'Hayır'],
        'Sinyal': [macd_sinyal],
        'Detay': [f"MACD: {macd.iloc[-1]:.2f}, Sinyal: {signal.iloc[-1]:.2f}"]
    }

# Keltner Kanal
def hesapla_keltner(df):
    df['ATR'] = df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min()
    df['Keltner_MA'] = df['Close'].rolling(window=20).mean()
    df['UpperKeltner'] = df['Keltner_MA'] + (df['ATR'] * 2)
    df['LowerKeltner'] = df['Keltner_MA'] - (df['ATR'] * 2)
    keltner_sinyal = 'Alım' if df['Close'].iloc[-1] < df['LowerKeltner'].iloc[-1] else 'Satım' if df['Close'].iloc[-1] > df['UpperKeltner'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['Keltner Kanal Kırılımı'],
        'Durum': ['Evet' if df['Close'].iloc[-1] < df['LowerKeltner'].iloc[-1] or df['Close'].iloc[-1] > df['UpperKeltner'].iloc[-1] else 'Hayır'],
        'Sinyal': [keltner_sinyal],
        'Detay': [f"Alt: {df['LowerKeltner'].iloc[-1]:.2f}, Üst: {df['UpperKeltner'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Kanaldan Çıkış
def hesapla_kanaldan_cikis(df):
    df['High20'] = df['High'].rolling(window=20).max()
    df['Low20'] = df['Low'].rolling(window=20).min()
    channel_breakout = df['Close'].iloc[-1] > df['High20'].iloc[-1] or df['Close'].iloc[-1] < df['Low20'].iloc[-1]
    channel_sinyal = 'Alım' if df['Close'].iloc[-1] > df['High20'].iloc[-1] else 'Satım' if df['Close'].iloc[-1] < df['Low20'].iloc[-1] else 'Yok'
    return {
        'Koşul': ['Kanaldan Çıkış'],
        'Durum': ['Evet' if channel_breakout else 'Hayır'],
        'Sinyal': [channel_sinyal],
        'Detay': [f"Yüksek: {df['High20'].iloc[-1]:.2f}, Düşük: {df['Low20'].iloc[-1]:.2f}, Bugün: {df['Close'].iloc[-1]:.2f}"]
    }

# Ana Strateji Fonksiyonu
@app.route('/analiz90', methods=['GET', 'POST'])
def sessiz_guc_stratejisi():
    try:
        # Parametreleri al (GET veya POST)
        if request.method == 'POST':
            data = request.get_json()
            hisse_kodu = data.get('hisse_kodu', 'TARKM.IS')
            days_back = int(data.get('days_back', 90))
            interval = data.get('interval', '1h')
            hacim_katsayisi = float(data.get('hacim_katsayisi', 1.1))
            tolerans = float(data.get('tolerans', 0.03))
        else:
            hisse_kodu = request.args.get('hisse_kodu', 'TARKM.IS')
            days_back = int(request.args.get('days_back', 90))
            interval = request.args.get('interval', '1h')
            hacim_katsayisi = float(request.args.get('hacim_katsayisi', 1.1))
            tolerans = float(request.args.get('tolerans', 0.03))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        df, uyari = veri_cek(hisse_kodu, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)

        required_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
        if df.empty or not all(col in df.columns for col in required_columns):
            return jsonify({'error': f"Gerekli sütunlar eksik veya veri boş: {df.columns}"}), 400
        if df[required_columns].isna().any().any():
            return jsonify({'error': "Eksik veri tespit edildi. Lütfen veri kaynağını kontrol edin."}), 400
        if len(df) < 50:
            return jsonify({'error': f"Yeterli veri yok ({len(df)} satır). En az 50 satır gerekli."}), 400

        df.index = df.index.tz_convert('Europe/Istanbul')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)

        current_price = float(df['Close'].iloc[-1])
        yesterday_price = float(df['Close'].iloc[-2])
        fiyat_degisim_yuzde = ((current_price - yesterday_price) / yesterday_price) * 100
        fiyat_uyari = f"⚠️ Dikkat: Fiyat %{fiyat_degisim_yuzde:.2f} düşüş gösterdi. Alım sinyalleri yanıltıcı olabilir!" if fiyat_degisim_yuzde < -5 else ""

        # Göstergeleri hesapla
        sessiz_guc_results = hesapla_sessiz_guc(df, hacim_katsayisi)
        hacim_artisi_results = hesapla_hacim_artisi(df, hacim_katsayisi)
        fhm_results = hesapla_fhm(df, hacim_katsayisi)
        phf_results = hesapla_phf(df, hacim_katsayisi)
        triangle_results = hesapla_triangle(df, tolerans=tolerans)
        double_top_results = hesapla_double_top(df, tolerans=tolerans)
        bearish_pennant_results = hesapla_bearish_pennant(df, tolerans=tolerans)
        bullish_flag_results = hesapla_bullish_flag(df, tolerans=tolerans)
        double_bottom_results = hesapla_double_bottom(df, tolerans=tolerans)
        destek_direnc_results = hesapla_destek_direnc(df, tolerans=tolerans)
        yukselen_kama_results = hesapla_yukselen_kama(df, tolerans=tolerans)
        kanal_yukari_results = hesapla_kanal_yukari(df, tolerans=tolerans)
        fincan_kulp_results = hesapla_fincan_kulp(df)
        ma5_22_results = hesapla_ma5_22(df)
        rsi_results = hesapla_rsi(df)
        ema_results = hesapla_ema50(df)
        sma_results = hesapla_sma50(df)
        wma_results = hesapla_wma50(df)
        fib_results = hesapla_fibonacci(df)
        bb_results = hesapla_bollinger(df)
        macd_results = hesapla_macd(df)
        keltner_results = hesapla_keltner(df)
        channel_results = hesapla_kanaldan_cikis(df)

        # Puanlama (ağırlıklı)
        puan_agirliklari = {
            'Sessiz Güç': 2, 'Hacim Artışı': 2, 'FHM': 2, 'PHF': 2, 'Triangle': 2,
            'Double Top': 2, 'Bearish Pennant': 1, 'Bullish Flag': 1, 'Double Bottom': 2,
            'Destek-Direnç': 2, 'Yükselen Kama': 2, 'Kanal Yukarı': 2,
            'Fincan Kulp': 1, 'MA5_22': 2, 'RSI': 3, 'EMA50': 2, 'SMA50': 2,
            'WMA50': 2, 'Fibonacci': 2, 'Bollinger': 2, 'MACD': 3, 'Keltner': 2,
            'Kanaldan Çıkış': 2
        }
        max_points = sum(puan_agirliklari.values())
        alim_puan = 0
        satim_puan = 0
        alim_puan_katsayisi = 0.5 if fiyat_degisim_yuzde < -5 else 1.0

        # Puanlama mantığı
        if any(sinyal == 'Alım' for sinyal in sessiz_guc_results['Sinyal']):
            alim_puan += puan_agirliklari['Sessiz Güç'] * alim_puan_katsayisi
        if hacim_artisi_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Hacim Artışı'] * alim_puan_katsayisi
        if any(sinyal == 'Alım' for sinyal in fhm_results['Sinyal']):
            alim_puan += puan_agirliklari['FHM'] * alim_puan_katsayisi
        if any(sinyal == 'Alım' for sinyal in phf_results['Sinyal']):
            alim_puan += puan_agirliklari['PHF'] * alim_puan_katsayisi
        if triangle_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Triangle'] * alim_puan_katsayisi
        elif triangle_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Triangle']
        if double_top_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Double Top']
        if bearish_pennant_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Bearish Pennant']
        if bullish_flag_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Bullish Flag'] * alim_puan_katsayisi
        if double_bottom_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Double Bottom'] * alim_puan_katsayisi
        if destek_direnc_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Destek-Direnç'] * alim_puan_katsayisi
        elif destek_direnc_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Destek-Direnç']
        if yukselen_kama_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Yükselen Kama']
        if kanal_yukari_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Kanal Yukarı'] * alim_puan_katsayisi
        elif kanal_yukari_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Kanal Yukarı']
        if fincan_kulp_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Fincan Kulp'] * alim_puan_katsayisi
        if ma5_22_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['MA5_22'] * alim_puan_katsayisi
        elif ma5_22_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['MA5_22']
        if rsi_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['RSI'] * alim_puan_katsayisi
        elif rsi_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['RSI']
        if ema_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['EMA50'] * alim_puan_katsayisi
        elif ema_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['EMA50']
        if sma_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['SMA50'] * alim_puan_katsayisi
        elif sma_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['SMA50']
        if wma_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['WMA50'] * alim_puan_katsayisi
        elif wma_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['WMA50']
        if fib_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Fibonacci'] * alim_puan_katsayisi
        if bb_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Bollinger'] * alim_puan_katsayisi
        elif bb_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Bollinger']
        if macd_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['MACD'] * alim_puan_katsayisi
        elif macd_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['MACD']
        if keltner_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Keltner'] * alim_puan_katsayisi
        elif keltner_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Keltner']
        if channel_results['Sinyal'][0] == 'Alım':
            alim_puan += puan_agirliklari['Kanaldan Çıkış'] * alim_puan_katsayisi
        elif channel_results['Sinyal'][0] == 'Satım':
            satim_puan += puan_agirliklari['Kanaldan Çıkış']

        sinyal_esigi = max_points * 0.25
        sinyal_turu = 'Sinyal Oluşmamış'
        if alim_puan > satim_puan and alim_puan > sinyal_esigi:
            sinyal_turu = 'Alım Sinyali'
        elif satim_puan > alim_puan and satim_puan > sinyal_esigi:
            sinyal_turu = 'Satım Sinyali'

        # Özet tablo
        ozet = pd.DataFrame({
            'Gösterge': [
                'Sessiz Güç', 'Hacim Artışı', 'FHM', 'PHF', 'Triangle', 'Double Top',
                'Bearish Pennant', 'Bullish Flag', 'Double Bottom', 'Destek-Direnç',
                'Yükselen Kama', 'Kanal Yukarı', 'Fincan Kulp', '5/22 MA', 'RSI',
                'EMA50', 'SMA50', 'WMA50', 'Fibonacci', 'Bollinger', 'MACD',
                'Keltner', 'Kanaldan Çıkış'
            ],
            'Sinyal': [
                'Alım' if any(sinyal == 'Alım' for sinyal in sessiz_guc_results['Sinyal']) else 'Yok',
                hacim_artisi_results['Sinyal'][0],
                'Alım' if any(sinyal == 'Alım' for sinyal in fhm_results['Sinyal']) else 'Yok',
                'Alım' if any(sinyal == 'Alım' for sinyal in phf_results['Sinyal']) else 'Yok',
                triangle_results['Sinyal'][0],
                double_top_results['Sinyal'][0],
                bearish_pennant_results['Sinyal'][0],
                bullish_flag_results['Sinyal'][0],
                double_bottom_results['Sinyal'][0],
                destek_direnc_results['Sinyal'][0],
                yukselen_kama_results['Sinyal'][0],
                kanal_yukari_results['Sinyal'][0],
                fincan_kulp_results['Sinyal'][0],
                ma5_22_results['Sinyal'][0],
                rsi_results['Sinyal'][0],
                ema_results['Sinyal'][0],
                sma_results['Sinyal'][0],
                wma_results['Sinyal'][0],
                fib_results['Sinyal'][0],
                bb_results['Sinyal'][0],
                macd_results['Sinyal'][0],
                keltner_results['Sinyal'][0],
                channel_results['Sinyal'][0]
            ]
        }).to_dict(orient='records')

        # JSON yanıtı oluştur
        response = {
            'hisse_kodu': hisse_kodu,
            'veri_tarihi': df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z'),
            'analiz_zamani': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'son_kapanis': current_price,
            'fiyat_degisim_yuzde': fiyat_degisim_yuzde,
            'fiyat_uyari': fiyat_uyari,
            'veri_uyari': uyari,
            'ozet': ozet,
            'sessiz_guc': sessiz_guc_results,
            'hacim_artisi': hacim_artisi_results,
            'fhm': fhm_results,
            'phf': phf_results,
            'triangle': triangle_results,
            'double_top': double_top_results,
            'bearish_pennant': bearish_pennant_results,
            'bullish_flag': bullish_flag_results,
            'double_bottom': double_bottom_results,
            'destek_direnc': destek_direnc_results,
            'yukselen_kama': yukselen_kama_results,
            'kanal_yukari': kanal_yukari_results,
            'fincan_kulp': fincan_kulp_results,
            'ma5_22': ma5_22_results,
            'rsi': rsi_results,
            'ema50': ema_results,
            'sma50': sma_results,
            'wma50': wma_results,
            'fibonacci': fib_results,
            'bollinger': bb_results,
            'macd': macd_results,
            'keltner': keltner_results,
            'kanaldan_cikis': channel_results,
            'nihai_sinyal': {
                'sinyal_turu': sinyal_turu,
                'alim_puan': alim_puan,
                'satim_puan': satim_puan,
                'max_puan': max_points
            },
            'uyari': [
                "Bu uygulama yalnızca teknik analiz araçlarıyla verileri yorumlamaktadır.",
                "YATIRIM TAVSİYESİ DEĞİLDİR.",
                "Nihai yatırım kararlarınızı, kendi araştırmalarınız doğrultusunda vermeniz önemlidir.",
                "Uygulama geliştiricisi, yapılan işlemlerden doğabilecek zararlardan sorumlu tutulamaz."
            ]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f"Hata oluştu: {str(e)}. Lütfen ticker sembolünü veya internet bağlantınızı kontrol edin."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)t lru_cache
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
