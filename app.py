from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# ---- TEKNİK ANALİZ FONKSİYONLARIN (direkt verdiğin kod) ----

# (Buraya az önce verdiğin TÜM kodun fonksiyonlarını direkt kopyaladım)
# (Kodu aşağıya uzatmamak için tekrar yazmadım, senin bana attığın kodun tamamını olduğu gibi buraya yapıştır.)
# Yani importlar dahil, bütün fonksiyonları (hesapla_sessiz_guc, hesapla_hacim_artisi, ... vb) en başa ekle!

# ---- ANA ANALİZ FONKSİYONU ----
def analiz90_full(hisse_kodu):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df = yf.download(hisse_kodu, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1h", auto_adjust=False)
        if df.empty:
            return {"hata": "Veri çekilemedi veya hisse kodu yanlış!"}
        # Aşağıda, ana analiz fonksiyonunu çağırıyoruz.
        # Tüm çıktıları dict olarak hazırlıyoruz.
        # Mevcut kodun direkt fonksiyonunu çağırıp bütün önemli çıktılarını JSON'a yazacağız.
        from io import StringIO
        import sys

        # Print çıktısını yakalamak için:
        old_stdout = sys.stdout
        mystdout = StringIO()
        sys.stdout = mystdout

        # Senin ana analiz fonksiyonunu çağırıyoruz:
        gosterge_listesi = ['Close', 'EMA50', 'SMA50', 'UpperBB', 'LowerBB', 'UpperKeltner', 'LowerKeltner']
        sessiz_guc_stratejisi(
            hisse_kodu,
            days_back=90,
            interval="1h",
            hacim_katsayisi=1.1,
            tolerans=0.03,
            gosterge_listesi=gosterge_listesi
        )

        # Print edilen tüm analiz sonuçlarını alıyoruz:
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        return {
            "hisse": hisse_kodu,
            "analiz": output
        }
    except Exception as e:
        return {"hata": str(e)}

@app.route('/analiz90', methods=['POST'])
def analiz90_api():
    data = request.get_json()
    hisse_kodu = data.get("hisse_kodu")
    if not hisse_kodu:
        return jsonify({"hata": "Hisse kodu eksik!"}), 400
    sonuc = analiz90_full(hisse_kodu)
    return jsonify(sonuc)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
