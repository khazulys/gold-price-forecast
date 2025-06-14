import os
import re
import numpy as np
import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from prettytable import PrettyTable


class EmasPredictor:
    def __init__(self, model_path="lstm_model_emas.h5", window=60, epochs=10, predict_days=7):
        self.model_path = model_path
        self.window = window
        self.epochs = epochs
        self.predict_days = predict_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.kurs = self.get_kurs_usd_idr()

    def get_kurs_usd_idr(self):
        """Ambil kurs USD ke IDR dari Google Finance."""
        url = "https://www.google.com/finance/quote/USD-IDR"
        res = cloudscraper.create_scraper().get(url)
        match = re.search(r'>(\d{1,3}(,\d{3})*(\.\d+)?|\d+\.\d+)<', res.text)
        if match:
            return float(match.group(1).replace(",", ""))
        raise ValueError("Gagal mengambil kurs USD-IDR")

    def get_historical_data(self):
        """Ambil data harga emas (USD) dari Yahoo Finance."""
        url = "https://finance.yahoo.com/quote/GLD/history"
        res = cloudscraper.create_scraper().get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.find_all("tr")
        data = []

        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) == 7:
                try:
                    close = cols[4].text.replace(",", "")
                    if "Dividend" not in close:
                        data.append(float(close))
                except Exception:
                    continue

        return pd.DataFrame(data[::-1], columns=["Close"])

    def get_local_gold_price(self):
        """Ambil harga emas batangan lokal dari logammulia.com"""
        url = "https://www.logammulia.com/id/harga-emas-hari-ini"
        res = cloudscraper.create_scraper().get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="table table-bordered")
        harga_dict = {}

        is_batangan = False
        for row in table.find_all("tr"):
            if "Emas Batangan" in row.text and not any(x in row.text for x in ["Gift", "Selamat", "Imlek", "Batik"]):
                is_batangan = True
                continue
            elif "Emas Batangan" in row.text and is_batangan:
                break
            if is_batangan:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    try:
                        gram = float(cols[0].text.replace("gr", "").strip())
                        harga = int(cols[2].text.strip().replace(",", "").replace(".", ""))
                        harga_dict[gram] = harga
                    except Exception:
                        continue
        return harga_dict

    def create_dataset(self, data):
        """Membuat dataset untuk pelatihan model LSTM."""
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:i+self.window])
            y.append(data[i+self.window])
        return np.array(X), np.array(y)

    def build_or_load_model(self, X, y):
        """Melatih model LSTM atau memuat model yang sudah ada."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False)
            self.model.compile(optimizer=Adam(), loss=MeanSquaredError())
        else:
            self.model = Sequential([
                LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
                Dense(1)
            ])
            self.model.compile(optimizer=Adam(), loss=MeanSquaredError())
            self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
            self.model.save(self.model_path)

    def forecast(self, scaled_data):
        """Melakukan prediksi harga emas untuk beberapa hari ke depan."""
        results = []
        seq = scaled_data[-self.window:].reshape(1, self.window, 1)

        for _ in range(self.predict_days):
            pred = self.model.predict(seq, verbose=0)[0]
            results.append(pred)
            seq = np.append(seq[:, 1:, :], [[pred]], axis=1)

        return self.scaler.inverse_transform(results)

    def run(self):
        # Ambil dan proses data historis
        df = self.get_historical_data()
        close_prices = df["Close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close_prices)
        X, y = self.create_dataset(scaled)

        # Bangun atau load model
        self.build_or_load_model(X, y)

        # Prediksi harga dalam IDR
        pred_usd = self.forecast(scaled)
        pred_idr = [float(x[0]) * self.kurs for x in pred_usd]

        # Ambil harga lokal
        harga_lokal = self.get_local_gold_price()

        # Hitung prediksi untuk 1 gram
        harga_now = harga_lokal.get(1.0)
        harga_pred = round(pred_idr[0])
        kenaikan = harga_pred - harga_now
        persentase = (kenaikan / harga_now) * 100

        print(f"Kurs USD-IDR: {self.kurs:,.2f}")
        print("\nPrediksi Harga Emas 1 Gram untuk 7 Hari ke Depan:")
        print(f"Harga Sekarang : Rp {harga_now:,}")
        print(f"Prediksi Harga : Rp {harga_pred:,}")
        print(f"Kenaikan       : Rp {kenaikan:,} ({persentase:.2f}%)")

        # Tampilkan tabel untuk semua ukuran
        print("\nPrediksi Semua Ukuran Emas Batangan:")
        table = PrettyTable()
        table.field_names = ["Gram", "Harga Sekarang", "Prediksi Harga", "Kenaikan"]

        for gram, hrg_now in sorted(harga_lokal.items()):
            pred = round(harga_pred * gram)
            naik = pred - hrg_now
            table.add_row([f"{gram} gr", f"Rp {hrg_now:,}", f"Rp {pred:,}", f"Rp {naik:,}"])

        print(table)


if __name__ == "__main__":
    app = EmasPredictor()
    app.run()