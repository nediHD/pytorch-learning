# -*- coding: utf-8 -*-
# ============================================================================
# EINFACHES RNN MODELL - FÜR GOOGLE COLAB (MIT GPU!)
# MULTI-SET TRAINING + LEARNING RATE SCHEDULER (PATIENCE=10)
# VERSION: RNN6 - Mit NIEDRIGEREM INITIAL LEARNING RATE (0.0005 statt 0.001)
# EXPERIMENTE: Ist ein konservativerer LR besser?
# ============================================================================

# SCHRITT 0: SETUP FÜR GOOGLE COLAB
print("=" * 80)
print("SCHRITT 0: SETUP FÜR GOOGLE COLAB - RNN6 MIT SCHEDULER (LR=0.0005)")
print("=" * 80)

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from pathlib import Path

# GPU-Verfügbarkeit prüfen
print(f"GPU verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU-Typ: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NICHT VERFÜGBAR! CPU wird verwendet (LANGSAM)")

# Für Colab - Dateien hochladen
print("\nDateien aus Colab hochladen:")
from google.colab import files
print("Klicke auf 'Choose Files' und wähle alle CSV-Dateien:")
print("  - ex_1.csv, ex_9.csv, ex_20.csv, ex_21.csv, ex_23.csv, ex_24.csv")

hochgeladene_dateien = files.upload()
print(f"\n✅ Hochgeladen: {list(hochgeladene_dateien.keys())}")
print(f"Anzahl der Dateien: {len(hochgeladene_dateien)}\n")

# ============================================================================
# COLAB VERSION - Direkt mit Dateien
# ============================================================================

print("\n" + "=" * 80)
print("SCHRITT 1: ALLE TRAININGSDATENSÄTZE LADEN")
print("=" * 80)

# Alle verfügbaren CSV-Dateien finden
print("Suche nach allen verfügbaren Trainingsdatensätzen aus data/train/...\n")

csv_dateien_verzeichnis = {}

# Versuche zuerst data/train/ (Standard-Speicherort)
trainingspfad = 'data/train'
if os.path.exists(trainingspfad):
    for datei in os.listdir(trainingspfad):
        if datei.endswith('.csv') and datei.startswith('ex_'):
            vollstaendiger_pfad = os.path.join(trainingspfad, datei)
            csv_dateien_verzeichnis[datei] = vollstaendiger_pfad
            print(f"  ✅ Gefunden: {datei}")

# Wenn data/train/ nicht existiert, suche im ganzen Verzeichnis
if not csv_dateien_verzeichnis:
    print("  Verzeichnis data/train/ nicht gefunden, suche anderswo...")
    for wurzel, verzeichnisse, dateien in os.walk('.'):
        for datei in dateien:
            if datei.endswith('.csv') and datei.startswith('ex_'):
                vollstaendiger_pfad = os.path.join(wurzel, datei)
                csv_dateien_verzeichnis[datei] = vollstaendiger_pfad
                print(f"  ✅ Gefunden: {datei}")

if not csv_dateien_verzeichnis:
    print("❌ Keine CSV-Dateien gefunden!")
    exit()

# Alle Dateien laden und kombinieren
print(f"\n📂 Kombiniere {len(csv_dateien_verzeichnis)} Dateien...")

alle_eingaben = []
alle_ausgaben = []
gesamt_zeilen = 0

for name, pfad in sorted(csv_dateien_verzeichnis.items()):
    daten = pd.read_csv(pfad)
    eingabe = daten['input_voltage'].values
    ausgabe = daten['el_power'].values

    alle_eingaben.append(eingabe)
    alle_ausgaben.append(ausgabe)
    gesamt_zeilen += len(daten)

    print(f"  ✅ {name}: {len(daten)} Zeilen (min={eingabe.min():.2f}V, max={eingabe.max():.2f}V)")

# Kombiniere alle in einen Array
eingangssignal = np.concatenate(alle_eingaben)
ausgangssignal = np.concatenate(alle_ausgaben)

print(f"\n📊 GESAMTER TRAININGSDATENSATZ:")
print(f"  Zeilen: {gesamt_zeilen}")
print(f"  Eingangssignal - Min: {eingangssignal.min():.2f}V, Max: {eingangssignal.max():.2f}V")
print(f"  Ausgangssignal - Min: {ausgangssignal.min():.2f}W, Max: {ausgangssignal.max():.2f}W")

print("\n" + "=" * 80)
print("SCHRITT 2: SEQUENZEN (N=451)")
print("=" * 80)

fenstergroesse = 451

eingabe_sequenzen = []
ausgabe_sequenzen = []

for i in range(len(eingangssignal) - fenstergroesse):
    eingabe_sequenzen.append(eingangssignal[i:i+fenstergroesse])
    ausgabe_sequenzen.append(ausgangssignal[i+fenstergroesse])

eingabe_sequenzen = np.array(eingabe_sequenzen)
ausgabe_sequenzen = np.array(ausgabe_sequenzen)

print(f"Es wurden {len(eingabe_sequenzen)} Sequenzen erstellt")

# Normalisierung
min_eingabe, max_eingabe = eingangssignal.min(), eingangssignal.max()
eingabe_sequenzen = (eingabe_sequenzen - min_eingabe) / (max_eingabe - min_eingabe)

min_ausgabe, max_ausgabe = ausgangssignal.min(), ausgangssignal.max()
ausgabe_sequenzen = (ausgabe_sequenzen - min_ausgabe) / (max_ausgabe - min_ausgabe)

print("Daten im Bereich 0-1 normalisiert")

print("\n" + "=" * 80)
print("SCHRITT 3: PYTORCH SETUP")
print("=" * 80)

eingabe_tensor = torch.FloatTensor(eingabe_sequenzen).unsqueeze(-1)
ausgabe_tensor = torch.FloatTensor(ausgabe_sequenzen).unsqueeze(-1)

datensatz = TensorDataset(eingabe_tensor, ausgabe_tensor)

chargengroesse = 32
datenlader = DataLoader(datensatz, batch_size=chargengroesse, shuffle=True)

print(f"Datensatz: {len(datensatz)} Beispiele")
print(f"Chargengröße: {chargengroesse}")
print(f"Anzahl der Chargen pro Epoche: {len(datenlader)}")

print("\n" + "=" * 80)
print("SCHRITT 4: RNN-MODELL")
print("=" * 80)

class EinfachesRNNModell(nn.Module):
    def __init__(self, eingabedimension=1, verstecktedimension=32, anzahl_lstm_schichten=3, ausgabedimension=1):
        super(EinfachesRNNModell, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.dichteSchicht = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        ausgabe_lstm1, _ = self.lstm1(x)
        ausgabe_lstm2, _ = self.lstm2(ausgabe_lstm1)
        ausgabe_lstm3, _ = self.lstm3(ausgabe_lstm2)
        letzter_wert = ausgabe_lstm3[:, -1, :]
        vorhersage = self.dichteSchicht(letzter_wert)
        return vorhersage

modell = EinfachesRNNModell()

# GPU!
geraet = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modell.to(geraet)
print(f"Modell auf: {geraet}")
if geraet.type == 'cuda':
    print("✅ GPU WIRD VERWENDET - Sollte SCHNELL sein! 🚀")

print("\n" + "=" * 80)
print("SCHRITT 5: VERLUSTFUNKTION")
print("=" * 80)

verlustfunktion = nn.MSELoss()
print("Verlustfunktion: MSE (Mean Squared Error)")

print("\n" + "=" * 80)
print("SCHRITT 6: TRAINING MIT EARLY STOPPING + LEARNING RATE SCHEDULER")
print("=" * 80)

# ← RNN6 EXPERIMENTE: NIEDRIGERE INITIAL LR (0.0005 statt 0.001)
optimizer = optim.Adam(modell.parameters(), lr=0.0005)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

anzahl_epochen = 300
bester_verlust = float('inf')
geduld_schwelle = 20
geduld_zaehler = 0
verlust_geschichte = []

print("🚀 Training gestartet (RNN6 mit LR=0.0005)...\n")

for epoche in tqdm(range(anzahl_epochen), desc="Epochen", unit="Epoche"):
    gesamt_verlust = 0
    anzahl_chargen = 0

    # Training Loop
    for eingabe_charge, ausgabe_charge in tqdm(datenlader, desc=f"Epoche {epoche+1}", leave=False):
        eingabe_charge = eingabe_charge.to(geraet)
        ausgabe_charge = ausgabe_charge.to(geraet)

        vorhersage = modell(eingabe_charge)
        verlust = verlustfunktion(vorhersage, ausgabe_charge)

        optimizer.zero_grad()
        verlust.backward()
        optimizer.step()

        gesamt_verlust += verlust.item()
        anzahl_chargen += 1

    durchschnittlicher_verlust = gesamt_verlust / anzahl_chargen
    verlust_geschichte.append(durchschnittlicher_verlust)

    # Ausgabe für jede Epoche
    print(f"✅ Epoche {epoche+1:3d}/{anzahl_epochen} | MSE Verlust: {durchschnittlicher_verlust:.6f}", end="")

    # Early Stopping Logik
    if durchschnittlicher_verlust < bester_verlust:
        bester_verlust = durchschnittlicher_verlust
        geduld_zaehler = 0
        torch.save(modell.state_dict(), 'einfachesRNN6.pth')
        print(f" | 💾 NEUES BEST-MODELL! (Geduld: 0/{geduld_schwelle})")
    else:
        geduld_zaehler += 1
        differenz = durchschnittlicher_verlust - bester_verlust
        print(f" | ⚠️ Keine Verbesserung (+{differenz:.6f}) | Geduld: {geduld_zaehler}/{geduld_schwelle}")

        if geduld_zaehler >= geduld_schwelle:
            print(f"\n🛑 EARLY STOPPING! Keine Verbesserung für {geduld_schwelle} Epochen!")
            print(f"Training beendet in Epoche {epoche+1}/{anzahl_epochen}")
            break

    scheduler.step(durchschnittlicher_verlust)

print(f"\n{'='*80}")
print(f"TRAINING ABGESCHLOSSEN!")
print(f"{'='*80}")
print(f"Bester Verlust: {bester_verlust:.6f}")
print(f"Gesamte Epochen: {len(verlust_geschichte)}/{anzahl_epochen}")
print(f"Zeitersparnis: {anzahl_epochen - len(verlust_geschichte)} Epochen übersprungen (Early Stopping)")

# Verbesserungsstatistik
if len(verlust_geschichte) > 1:
    anfangs_verlust = verlust_geschichte[0]
    verbesserungs_prozentsatz = ((anfangs_verlust - bester_verlust) / anfangs_verlust) * 100
    print(f"Verbesserung: {anfangs_verlust:.6f} → {bester_verlust:.6f} ({verbesserungs_prozentsatz:.1f}%)")

print("\n" + "=" * 80)
print("SCHRITT 7: MODELL HERUNTERLADEN")
print("=" * 80)

# Modell von Colab herunterladen
files.download('einfachesRNN6.pth')
print("✅ Modell 'einfachesRNN6.pth' heruntergeladen!")

print("\n" + "=" * 80)
print("FERTIG!")
print("=" * 80)
print("Modell ist trainiert und heruntergeladen!")
print("\n🎉 RNN6 (Niedrigere LR=0.0005) ist abgeschlossen!")
print("Vergleiche RMSE mit RNN2 und RNN3 um zu sehen welcher LR am besten passt!")
