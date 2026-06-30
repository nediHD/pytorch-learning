# ============================================================================
# EINFACHES RNN MODELL - ZA GOOGLE COLAB (SA GPU!)
# Nije RNN sa LSTM i MSE Verlustfunktion
# ============================================================================

# KORAK 0: SETUP ZA GOOGLE COLAB
print("=" * 80)
print("KORAK 0: SETUP ZA GOOGLE COLAB")
print("=" * 80)

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # Za progress bar

# Provjeri GPU
print(f"GPU dostupan: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU tip: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NIJE DOSTUPAN! Koristi CPU (SPORO)")

# Za Colab - učitaj datoteke
print("\nUčitavanje datoteka iz Colab:")
from google.colab import files
print("Klikni na 'Choose Files' i odaberi:")
print("  - data/train/ex_1.csv")
print("  - data/test/ex_*.csv (ako imaš)")

uploaded = files.upload()
print(f"\n✅ Učitano: {list(uploaded.keys())}")
print(f"Broj datoteka: {len(uploaded)}\n")

# ============================================================================
# KOLAB VERZIJA - Direktno sa datotekama
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 1: UČITAVANJE PODATAKA")
print("=" * 80)

# Pronađi datoteku automatski
import os
from pathlib import Path

print("Tražim ex_1.csv...")
csv_file = None

# Provjeri sve moguće lokacije
for root, dirs, csv_files in os.walk('.'):
    if 'ex_1.csv' in csv_files:
        csv_file = os.path.join(root, 'ex_1.csv')
        print(f"✅ Pronađena: {csv_file}")
        break

if csv_file is None:
    print("❌ ex_1.csv nije pronađena!")
    print("\nDatoteke u trenutnom direktoriju:")
    for root, dirs, csv_files in os.walk('.'):
        for file in csv_files:
            if file.endswith('.csv'):
                print(f"  {os.path.join(root, file)}")
    exit()

# Učitaj datoteku
print(f"📂 TRAINING datoteka: {csv_file}")
daten = pd.read_csv(csv_file)
print(f"✅ Učitano: {len(daten)} redaka\n")

# Ekstrahiraj signale
eingangssignal = daten['input_voltage'].values
ausgangssignal = daten['el_power'].values

print(f"Ulazni signal - Min: {eingangssignal.min():.2f}V, Max: {eingangssignal.max():.2f}V")
print(f"Izlazni signal - Min: {ausgangssignal.min():.2f}W, Max: {ausgangssignal.max():.2f}W")

print("\n" + "=" * 80)
print("KORAK 2: SEKVENCE (N=451)")
print("=" * 80)

N = 451

eingabeSequenzen = []
ausgabeSequenzen = []

for i in range(len(eingangssignal) - N):
    eingabeSequenzen.append(eingangssignal[i:i+N])
    ausgabeSequenzen.append(ausgangssignal[i+N])

eingabeSequenzen = np.array(eingabeSequenzen)
ausgabeSequenzen = np.array(ausgabeSequenzen)

print(f"Kreirano {len(eingabeSequenzen)} sekvenci")

# Normalizacija
minEingabe, maxEingabe = eingangssignal.min(), eingangssignal.max()
eingabeSequenzen = (eingabeSequenzen - minEingabe) / (maxEingabe - minEingabe)

minAusgabe, maxAusgabe = ausgangssignal.min(), ausgangssignal.max()
ausgabeSequenzen = (ausgabeSequenzen - minAusgabe) / (maxAusgabe - minAusgabe)

print("Podaci normalizirani")

print("\n" + "=" * 80)
print("KORAK 3: PYTORCH SETUP")
print("=" * 80)

eingabeSequenzenTensor = torch.FloatTensor(eingabeSequenzen).unsqueeze(-1)
ausgabeSequenzenTensor = torch.FloatTensor(ausgabeSequenzen).unsqueeze(-1)

dataset = TensorDataset(eingabeSequenzenTensor, ausgabeSequenzenTensor)

chargengröße = 32
datenlader = DataLoader(dataset, batch_size=chargengröße, shuffle=True)

print(f"Dataset: {len(dataset)} primjera")
print(f"Batches po epohi: {len(datenlader)}")

print("\n" + "=" * 80)
print("KORAK 4: RNN MODELL")
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

modell = EinfachesRNNModell()

# GPU!
gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modell.to(gerät)
print(f"Model na: {gerät}")
if gerät.type == 'cuda':
    print("✅ GPU KORIŠTEN - Trebao bi biti BRZI! 🚀")

print("\n" + "=" * 80)
print("KORAK 5: LOSS FUNKCIJA")
print("=" * 80)

verlustfunktion = nn.MSELoss()
print("Verlustfunktion: MSE")

print("\n" + "=" * 80)
print("KORAK 6: TRENIRANJE (300 EPOHA)")
print("=" * 80)

optimierer = optim.Adam(modell.parameters(), lr=0.001)

anzahlEpochen = 300
besterVerlust = float('inf')

print("🚀 Treniranje početo... (ovo može potrajati)\n")

for epoche in tqdm(range(anzahlEpochen), desc="Epohe", unit="epoha"):
    gesamtVerlust = 0
    anzahlBatches = 0

    for XCharge, yCharge in tqdm(datenlader, desc=f"Epoha {epoche+1}", leave=False):
        XCharge = XCharge.to(gerät)
        yCharge = yCharge.to(gerät)

        yVorhersage = modell(XCharge)
        loss = verlustfunktion(yVorhersage, yCharge)

        optimierer.zero_grad()
        loss.backward()
        optimierer.step()

        gesamtVerlust += loss.item()
        anzahlBatches += 1

    durchschnittlicherVerlust = gesamtVerlust / anzahlBatches

    # Ispis SVAKE epohe
    print(f"✅ Epoha {epoche+1:3d}/{anzahlEpochen} | MSE Verlust: {durchschnittlicherVerlust:.6f}")

    if durchschnittlicherVerlust < besterVerlust:
        besterVerlust = durchschnittlicherVerlust
        torch.save(modell.state_dict(), 'einfachesRNN.pth')
        print(f"   💾 Novi najbolji model spremeljen! Verlust: {durchschnittlicherVerlust:.6f}")

print(f"\nTreniranje završeno!")
print(f"Najbolji Verlust: {besterVerlust:.6f}")

print("\n" + "=" * 80)
print("KORAK 7: PREUZIMANJE MODELA")
print("=" * 80)

# Preuzmi model sa Colab-a
files.download('einfachesRNN.pth')
print("✅ Model 'einfachesRNN.pth' preuzet!")

print("\n" + "=" * 80)
print("GOTOVO!")
print("=" * 80)
print("Model je treniran i preuzet!")
