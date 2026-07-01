# -*- coding: utf-8 -*-
# ============================================================================
# EINFACHES RNN MODELL - ZA GOOGLE COLAB (SA GPU!)
# MULTI-SET TRAINING: ex_1, ex_4, ex_22 + EARLY STOPPING
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
from tqdm import tqdm

# Provjeri GPU
print(f"GPU dostupan: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU tip: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NIJE DOSTUPAN! Koristi CPU (SPORO)")

# Za Colab - učitaj datoteke
print("\nUčitavanje datoteka iz Colab:")
from google.colab import files
print("Klikni na 'Choose Files' i odaberi sve CSV datoteke:")
print("  - ex_1.csv, ex_4.csv, ex_22.csv (svi dostupni setovi)")

uploaded = files.upload()
print(f"\n✅ Učitano: {list(uploaded.keys())}")
print(f"Broj datoteka: {len(uploaded)}\n")

# ============================================================================
# KOLAB VERZIJA - Direktno sa datotekama
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 1: UČITAVANJE SVIH TRAINING SETOVA")
print("=" * 80)

import os
from pathlib import Path

# Pronađi sve dostupne CSV datoteke iz data/train/
print("Tražim sve dostupne training setove iz data/train/...\n")

csv_files_dict = {}

# Prvo pokušaj data/train/ (standardna lokacija)
train_path = 'data/train'
if os.path.exists(train_path):
    for file in os.listdir(train_path):
        if file.endswith('.csv') and file.startswith('ex_'):
            full_path = os.path.join(train_path, file)
            csv_files_dict[file] = full_path
            print(f"  ✅ Pronađena: {file}")

# Ako nema data/train/, traži u cijelom direktoriju
if not csv_files_dict:
    print("  Direktorij data/train/ nije pronađen, tražim drugdje...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv') and file.startswith('ex_'):
                full_path = os.path.join(root, file)
                csv_files_dict[file] = full_path
                print(f"  ✅ Pronađena: {file}")

if not csv_files_dict:
    print("❌ Nema CSV datoteka!")
    exit()

# Učitaj i objedini sve datoteke
print(f"\n📂 Objedinjujem {len(csv_files_dict)} datoteke...")

svi_ulazi = []
svi_izlazi = []
ukupni_redci = 0

for naziv, putanja in sorted(csv_files_dict.items()):
    daten = pd.read_csv(putanja)
    ulaz = daten['input_voltage'].values
    izlaz = daten['el_power'].values

    svi_ulazi.append(ulaz)
    svi_izlazi.append(izlaz)
    ukupni_redci += len(daten)

    print(f"  ✅ {naziv}: {len(daten)} redaka (min={ulaz.min():.2f}V, max={ulaz.max():.2f}V)")

# Objedini sve u jedan niz
eingangssignal = np.concatenate(svi_ulazi)
ausgangssignal = np.concatenate(svi_izlazi)

print(f"\n📊 UKUPNI TRAINING SET:")
print(f"  Redaka: {ukupni_redci}")
print(f"  Ulazni signal - Min: {eingangssignal.min():.2f}V, Max: {eingangssignal.max():.2f}V")
print(f"  Izlazni signal - Min: {ausgangssignal.min():.2f}W, Max: {ausgangssignal.max():.2f}W")

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
print("KORAK 6: TRENIRANJE SA EARLY STOPPING")
print("=" * 80)

optimierer = optim.Adam(modell.parameters(), lr=0.001)

anzahlEpochen = 300
besterVerlust = float('inf')
earlyStopping_patience = 20  # Zaustavi ako se loss ne poboljša 20 epoha
earlyStopping_counter = 0
verlust_historija = []

print("🚀 Treniranje početo (Early Stopping aktivno - patience=20)...\n")

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
    verlust_historija.append(durchschnittlicherVerlust)

    # Ispis SVAKE epohe
    print(f"✅ Epoha {epoche+1:3d}/{anzahlEpochen} | MSE Verlust: {durchschnittlicherVerlust:.6f}", end="")

    # Early Stopping logika
    if durchschnittlicherVerlust < besterVerlust:
        besterVerlust = durchschnittlicherVerlust
        earlyStopping_counter = 0  # Resetiraj counter
        torch.save(modell.state_dict(), 'einfachesRNN.pth')
        print(f" | 💾 NOVI NAJBOLJI! (patience: 0/{earlyStopping_patience})")
    else:
        earlyStopping_counter += 1
        razlika = durchschnittlicherVerlust - besterVerlust
        print(f" | ⚠️ Nema poboljšanja (+{razlika:.6f}) | patience: {earlyStopping_counter}/{earlyStopping_patience}")

        # Zaustavi treniranje ako nema poboljšanja
        if earlyStopping_counter >= earlyStopping_patience:
            print(f"\n🛑 EARLY STOPPING! Nema poboljšanja {earlyStopping_patience} epoha!")
            print(f"Treniranje prekinuto u epohi {epoche+1}/{anzahlEpochen}")
            break

print(f"\n{'='*80}")
print(f"TRENIRANJE ZAVRŠENO!")
print(f"{'='*80}")
print(f"Najbolji Verlust: {besterVerlust:.6f}")
print(f"Ukupne epohe: {len(verlust_historija)}/{anzahlEpochen}")
print(f"Ušteda vremena: {anzahlEpochen - len(verlust_historija)} epoha preskočeno (Early Stopping)")

# Statisitka poboljšanja
if len(verlust_historija) > 1:
    pocetni_verlust = verlust_historija[0]
    poboljsanje_procenta = ((pocetni_verlust - besterVerlust) / pocetni_verlust) * 100
    print(f"Poboljšanje: {pocetni_verlust:.6f} → {besterVerlust:.6f} ({poboljsanje_procenta:.1f}%)")

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
