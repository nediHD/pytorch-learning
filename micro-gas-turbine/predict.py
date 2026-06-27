"""
Ein trainiertes Modell laden und auf Testdaten Vorhersagen machen.

Dies zeigt wie man ein gespeichertes Modell in der Praxis wiederverwenden kann.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

from model import WindowMLP


def predict_on_test_data():
    """
    Lädt das trainierte Modell und macht Vorhersagen auf Test-Daten.
    """

    # ============================================================
    # 1. CHECKPOINT LADEN
    # ============================================================
    print("=== 1. Loading Checkpoint ===\n")

    model_path = 'saved_models/model.pth'

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} nicht gefunden!")
        print("Bitte erst 'python train.py' ausführen.\n")
        return

    # Lade den Checkpoint (mit weights_only=False für numpy-kompatibilität)
    # PyTorch 2.6+ bevorzugt weights_only=True, aber das funktioniert noch nicht mit
    # numpy-Skalaren in gespeicherten Dicts. Wir vertrauen unserer eigenen Datei.
    checkpoint = torch.load(model_path, weights_only=False)

    window_size = checkpoint['window_size']
    input_min = checkpoint['input_min']
    input_max = checkpoint['input_max']
    target_min = checkpoint['target_min']
    target_max = checkpoint['target_max']

    print(f"Loaded checkpoint:")
    print(f"  Window size: {window_size}")
    print(f"  Input range: [{input_min:.2f}, {input_max:.2f}]")
    print(f"  Target range: [{target_min:.2f}, {target_max:.2f}]\n")

    # ============================================================
    # 2. MODELL REKONSTRUIEREN UND GEWICHTE LADEN
    # ============================================================
    print("=== 2. Reconstructing Model ===\n")

    model = WindowMLP(window_size=window_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # WICHTIG: Evaluation Mode

    print("Model loaded and set to evaluation mode\n")

    # model.eval() ist kritisch. Es schaltet Dropout aus und setzt Batch Normalization
    # in den Inference-Modus. Ohne eval() macht das Modell Vorhersagen mit zufälligen
    # Dropout-Maskierungen, was zu unterschiedlichen Ergebnissen jedes Mal führt.

    # ============================================================
    # 3. TEST-DATEN LADEN
    # ============================================================
    print("=== 3. Loading Test Data ===\n")

    test_dir = 'data/test'
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])

    all_test_data = []
    for test_file in test_files:
        df = pd.read_csv(os.path.join(test_dir, test_file))
        all_test_data.append(df)
        print(f"  Loaded {test_file}: {len(df)} rows")

    full_test_data = pd.concat(all_test_data, ignore_index=True)
    print(f"  Total test data: {len(full_test_data)} rows\n")

    # ============================================================
    # 4. DATEN NORMALISIEREN
    # ============================================================
    print("=== 4. Normalizing Test Data ===\n")

    # WICHTIG: Verwende die TRAININGSDATEN-Normalisierungswerte, NICHT neu berechnen!
    # Warum? Wenn wir neu mit den Test-Daten normalisieren, ist das Data Leakage.
    # Das Modell würde "sehen" wie die Test-Daten verteilt sind.

    time_diff = full_test_data['time'].diff().fillna(1.0).values
    input_voltage = full_test_data['input_voltage'].values.astype(np.float32)
    el_power = full_test_data['el_power'].values.astype(np.float32)

    # Normalisiere mit Training-Min/Max
    input_voltage_norm = (input_voltage - input_min) / (input_max - input_min + 1e-8)
    el_power_norm = (el_power - target_min) / (target_max - target_min + 1e-8)

    # ============================================================
    # 5. SLIDING WINDOWS ERSTELLEN
    # ============================================================
    print("=== 5. Creating Windows ===\n")

    X_test = []
    y_test_actual = []

    for i in range(len(full_test_data) - window_size):
        window_time_delta = time_diff[i:i+window_size]
        window_input = input_voltage_norm[i:i+window_size]

        window = np.stack([window_time_delta, window_input], axis=1).astype(np.float32)
        X_test.append(window)
        y_test_actual.append(el_power_norm[i+window_size])

    X_test = np.array(X_test)
    y_test_actual = np.array(y_test_actual)

    print(f"Created {len(X_test)} test windows\n")

    # ============================================================
    # 6. INFERENZ
    # ============================================================
    print("=== 6. Running Inference ===\n")

    # torch.no_grad(): Deaktiviere Autograd für Speicher und Geschwindigkeit
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.reshape(len(X_test), -1), dtype=torch.float32)
        predictions_norm = model(X_test_tensor).squeeze().numpy()

    print(f"Made predictions for {len(predictions_norm)} samples\n")

    # ============================================================
    # 7. DENORMALISIERUNG
    # ============================================================
    print("=== 7. Denormalizing Predictions ===\n")

    predictions = predictions_norm * (target_max - target_min) + target_min
    actuals = y_test_actual * (target_max - target_min) + target_min

    # ============================================================
    # 8. ERGEBNISSE ANZEIGEN
    # ============================================================
    print("=== 8. Results ===\n")

    # Berechne RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))

    print(f"RMSE: {rmse:.2f} watts")
    print(f"MAE:  {mae:.2f} watts\n")

    # Zeige erste 20 Vorhersagen vs. tatsächliche Werte
    print("First 20 Predictions vs Actual:\n")
    print(f"{'Index':<8} {'Predicted (W)':<18} {'Actual (W)':<18} {'Error (W)':<12}")
    print("-" * 60)

    for i in range(min(20, len(predictions))):
        error = predictions[i] - actuals[i]
        print(f"{i:<8} {predictions[i]:<18.2f} {actuals[i]:<18.2f} {error:<12.2f}")

    print()

    # ============================================================
    # 9. STATISTIKEN
    # ============================================================
    print("=== 9. Statistics ===\n")

    max_error = np.max(np.abs(predictions - actuals))
    min_error = np.min(np.abs(predictions - actuals))

    print(f"Max absolute error: {max_error:.2f} watts")
    print(f"Min absolute error: {min_error:.2f} watts")
    print(f"Mean absolute error: {mae:.2f} watts")
    print(f"Std of errors: {np.std(predictions - actuals):.2f} watts\n")


if __name__ == '__main__':
    predict_on_test_data()

# === NÄCHSTE SCHRITTE für Lernende ===
# 1. Plotte Predictions vs Actual über die Zeit (matplotlib):
#    import matplotlib.pyplot as plt
#    plt.plot(actuals, label='Actual')
#    plt.plot(predictions, label='Predicted')
#    plt.legend()
#    plt.show()
#
# 2. Versuche, mit den Training-Daten vorherzusagen (nicht Test).
#    Sollte viel besser sein — warum? (Hint: Overfitting)
#
# 3. Ändere den Window Size in train.py, trainiere neu, und vergleiche RMSE.
