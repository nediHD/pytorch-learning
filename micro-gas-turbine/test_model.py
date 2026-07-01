# -*- coding: utf-8 -*-
# ============================================================================
# TEST MODELA - EVALUACIJA NA ex_4.csv i ex_22.csv
# ============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("TEST MODELA NA ex_4 I ex_22")
print("=" * 80)

# ============================================================================
# KORAK 1: UČITAJ TRENIRANI MODEL
# ============================================================================

print("\nKORAK 1: UČITAVANJE MODELA")
print("=" * 80)

class EinfachesRNNModell(nn.Module):
    def __init__(self, eingabegröße=1, verstecktgröße=32, anzahlLSTMSchichten=3, ausgabegröße=1):
        super(EinfachesRNNModell, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.dichteSchicht = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        ausgabeLSTM1, _ = self.lstm1(x)
        ausgabeLSTM2, _ = self.lstm2(ausgabeLSTM1)
        ausgabeLSTM3, _ = self.lstm3(ausgabeLSTM2)
        letzterWert = ausgabeLSTM3[:, -1, :]
        vorhersage = self.dichteSchicht(letzterWert)
        return vorhersage

# Učitaj model
modell = EinfachesRNNModell()
gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modell.to(gerät)

try:
    modell.load_state_dict(torch.load('einfachesRNN.pth', map_location=gerät))
    print("✅ Model učitan: einfachesRNN.pth")
except FileNotFoundError:
    print("❌ Model nije pronađen!")
    exit()

modell.eval()

# ============================================================================
# KORAK 2: UČITAJ TRAINING PODATKE (za normalizacijske parametre)
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 2: UČITAVANJE TRAINING PODATAKA")
print("=" * 80)

skriptpfad = Path(__file__).parent
csvdatei_train = skriptpfad / 'data' / 'train' / 'ex_1.csv'

daten_train = pd.read_csv(csvdatei_train)
eingangssignal_train = daten_train['input_voltage'].values
ausgangssignal_train = daten_train['el_power'].values

# SPREMANJE NORMALIZACIJSKIH PARAMETARA
minEingabe = eingangssignal_train.min()
maxEingabe = eingangssignal_train.max()
minAusgabe = ausgangssignal_train.min()
maxAusgabe = ausgangssignal_train.max()

print(f"📊 Training podaci normalizacijski parametri:")
print(f"   Ulaz: min={minEingabe:.2f}V, max={maxEingabe:.2f}V")
print(f"   Izlaz: min={minAusgabe:.2f}W, max={maxAusgabe:.2f}W")

# ============================================================================
# KORAK 3: UČITAJ TEST DATOTEKE (ex_4 i ex_22)
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 3: UČITAVANJE TEST DATOTEKA")
print("=" * 80)

N = 451

test_datoteke = {
    'ex_4.csv': skriptpfad / 'data' / 'ex_4.csv',
    'ex_22.csv': skriptpfad / 'data' / 'ex_22.csv'
}

rezultati = {}

for naziv, putanja in test_datoteke.items():
    if not putanja.exists():
        print(f"❌ {naziv} nije pronađena na {putanja}")
        continue

    print(f"\n📂 Testiram: {naziv}")

    # Učitaj datoteku
    daten_test = pd.read_csv(putanja)
    ulaz_test = daten_test['input_voltage'].values
    izlaz_test = daten_test['el_power'].values

    print(f"   Redaka: {len(daten_test)}")
    print(f"   Ulaz: min={ulaz_test.min():.2f}V, max={ulaz_test.max():.2f}V")
    print(f"   Izlaz: min={izlaz_test.min():.2f}W, max={izlaz_test.max():.2f}W")

    # ============================================================================
    # KORAK 4: KREIRAJ SEKVENCE
    # ============================================================================

    ulazSekvence = []
    izlazSekvence = []

    for i in range(len(ulaz_test) - N):
        ulazSekvence.append(ulaz_test[i:i+N])
        izlazSekvence.append(izlaz_test[i+N])

    ulazSekvence = np.array(ulazSekvence)
    izlazSekvence = np.array(izlazSekvence)

    # Normalizacija sa TRAINING parametrima
    ulazSekvence = (ulazSekvence - minEingabe) / (maxEingabe - minEingabe)
    izlazSekvence = (izlazSekvence - minAusgabe) / (maxAusgabe - minAusgabe)

    # ============================================================================
    # KORAK 5: NAPRAVIT PREDVIĐANJA
    # ============================================================================

    ulazTensor = torch.FloatTensor(ulazSekvence).unsqueeze(-1).to(gerät)
    izlazTensor = torch.FloatTensor(izlazSekvence).unsqueeze(-1).to(gerät)

    with torch.no_grad():
        predviđanja = modell(ulazTensor)
        loss = nn.MSELoss()(predviđanja, izlazTensor)

    # Denormalizacija
    predviđanja_denorm = predviđanja.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe
    izlaz_denorm = izlazTensor.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe

    # RMSE izračun
    RMSE = np.sqrt(np.mean((predviđanja_denorm - izlaz_denorm) ** 2))
    MAE = np.mean(np.abs(predviđanja_denorm - izlaz_denorm))

    rezultati[naziv] = {
        'MSE': loss.item(),
        'RMSE': RMSE,
        'MAE': MAE,
        'sekvenci': len(ulazSekvence)
    }

    print(f"\n   📊 REZULTATI:")
    print(f"      Sekvenci: {len(ulazSekvence)}")
    print(f"      MSE: {loss.item():.6f}")
    print(f"      RMSE: {RMSE:.2f} W")
    print(f"      MAE: {MAE:.2f} W")

# ============================================================================
# KORAK 6: ÖSSZEHASONLÍTÁS
# ============================================================================

print("\n" + "=" * 80)
print("POREĐENJE REZULTATA")
print("=" * 80)

for naziv, metrike in rezultati.items():
    print(f"\n{naziv}:")
    print(f"  RMSE: {metrike['RMSE']:.2f} W")
    print(f"  MAE:  {metrike['MAE']:.2f} W")
    print(f"  MSE:  {metrike['MSE']:.6f}")

print("\n" + "=" * 80)
print("TEST ZAVRŠEN!")
print("=" * 80)
