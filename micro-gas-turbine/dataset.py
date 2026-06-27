"""
PyTorch Dataset für Micro Gas Turbine Zeitreihen-Regression.

Konzept:
- Lädt alle CSV-Dateien aus einem Ordner
- Erstellt ein Sliding Window: X = [time_delta, input_voltage] über die letzten WINDOW_SIZE Schritte
- y = el_power am Ende des Fensters
- Normalisiert alle Werte per Min-Max (speichert diese Werte für predict.py)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class GasTurbineDataset(Dataset):
    """
    Ein PyTorch Dataset, das Mikrogasturbinen-Zeitreihen mit Sliding Window lädt.

    Ein Dataset muss drei Methoden haben:
    - __init__: Laden und Vorbereitung der Daten
    - __len__: Anzahl der Samples (hier: Anzahl der Windows)
    - __getitem__: Ein einzelnes Sample (X, y) zurückgeben
    """

    def __init__(self, data_dir, window_size=30):
        """
        Args:
            data_dir (str): Pfad zum Ordner mit CSV-Dateien
            window_size (int): Größe des Sliding Windows (Anzahl der Zeitschritte)
        """
        self.window_size = window_size

        # Schritt 1: Alle CSVs laden und zusammenführen
        all_data = []
        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

        print(f"Lade Dateien aus {data_dir}: {csv_files}")

        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            all_data.append(df)
            print(f"  - {csv_file}: {len(df)} Zeilen")

        # Alle Daten zusammenführen
        full_data = pd.concat(all_data, ignore_index=True)
        print(f"Gesamt: {len(full_data)} Zeilen\n")

        # Schritt 2: time_delta berechnen (Differenz zwischen aufeinanderfolgenden Timestamps)
        # Das ist informativer als die rohe Zeit, da verschiedene Experimente zu verschiedenen
        # absoluten Zeiten starten.
        time_diff = full_data['time'].diff().fillna(1.0).values  # fillna(1.0) für den ersten Wert

        # Schritt 3: Features und Target extrahieren
        input_voltage = full_data['input_voltage'].values.astype(np.float32)
        el_power = full_data['el_power'].values.astype(np.float32)

        # Schritt 4: Min-Max Normalisierung (für später: in predict.py verwenden wir diese Werte)
        self.input_min = input_voltage.min()
        self.input_max = input_voltage.max()
        self.target_min = el_power.min()
        self.target_max = el_power.max()

        input_voltage_norm = (input_voltage - self.input_min) / (self.input_max - self.input_min + 1e-8)
        el_power_norm = (el_power - self.target_min) / (self.target_max - self.target_min + 1e-8)

        # Schritt 5: Sliding Window erzeugen
        # X: [time_delta, input_voltage] für jedes Fenster
        # y: el_power am Ende des Fensters
        self.X = []
        self.y = []

        for i in range(len(full_data) - window_size):
            window_time_delta = time_diff[i:i+window_size]
            window_input = input_voltage_norm[i:i+window_size]

            # Stack zu (window_size, 2)
            window = np.stack([window_time_delta, window_input], axis=1).astype(np.float32)
            self.X.append(window)

            # Target ist die Leistung am ENDE des Fensters
            self.y.append(el_power_norm[i+window_size])

        print(f"Sliding Windows erstellt: {len(self.X)} samples mit window_size={window_size}\n")

    def __len__(self):
        """Gibt die Anzahl der verfügbaren Samples zurück."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Gibt ein Sample (X, y) als PyTorch Tensoren zurück.

        PyTorch wird diese Methode aufrufen, wenn wir über den DataLoader iterieren.
        """
        # Flatten: (window_size, 2) → (window_size * 2,)
        X = torch.tensor(self.X[idx].flatten(), dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

    def denormalize_target(self, normalized_value):
        """
        Konvertiert einen normalisierten Leistungswert zurück in Watt.

        Wird in predict.py verwendet, um Vorhersagen in Original-Einheiten zu zeigen.
        """
        return normalized_value * (self.target_max - self.target_min) + self.target_min


# === NÄCHSTE SCHRITTE für Lernende ===
# 1. Versuche WINDOW_SIZE zu ändern (z.B. auf 50 oder 10) — wie ändert sich die Modellperformance?
# 2. Füge ein drittes Feature hinzu: die rohe Zeit seit Experiment-Start (zusätzlich zu time_delta)
# 3. Untersuche ein einzelnes Sample: dataset[0][0].shape — welche Form hat das Fenster?
