# -*- coding: utf-8 -*-
# ============================================================================
# VIZUALIZACIJA MODELA - PROGNOZA VS STVARNO
# Odvojena po modelima i datasetima
# ============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Za serverske okoline

print("=" * 80)
print("VIZUALIZACIJA MODELA - SLIKE PROGNOZA")
print("=" * 80)

# ============================================================================
# KORAK 1: DEFINICIJA MODELA
# ============================================================================

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

# ============================================================================
# KORAK 2: UČITAJ MODELE
# ============================================================================

print("\nKORAK 1: UČITAVANJE MODELA")
print("=" * 80)

gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modeli_za_testiranje = {
    '../RNN/models/einfachesRNN.pth': 'Original (ex_1 samo)',
    '../RNN1/models/einfachesRNN1.pth': 'Multi-Set + Early Stopping',
    '../RNN2/models/einfachesRNN2.pth': 'Multi-Set + Scheduler (LR=0.001)',
    '../RNN3/models/einfachesRNN3.pth': 'Multi-Set + Scheduler (LR=0.01 - Viši)',
    '../RNN4/models/einfachesRNN4.pth': 'Multi-Set + Scheduler (4 LSTM - Dublja)',
    '../RNN5/models/einfachesRNN5.pth': 'Multi-Set + Scheduler (2 LSTM - Plića)',
    '../RNN6/models/einfachesRNN6.pth': 'Multi-Set + Scheduler (LR=0.0005 - Niži)'
}

ucitani_modeli = {}

for model_putanja, opis in modeli_za_testiranje.items():
    try:
        modell = EinfachesRNNModell()
        modell.to(gerät)
        modell.load_state_dict(torch.load(model_putanja, map_location=gerät))
        modell.eval()
        ucitani_modeli[model_putanja] = modell
        print(f"✅ Učitan: {model_putanja} ({opis})")
    except FileNotFoundError:
        print(f"❌ Model nije pronađen: {model_putanja}")
        continue

# ============================================================================
# KORAK 3: UČITAJ TRAINING PODATKE
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 2: UČITAVANJE TRAINING PODATAKA")
print("=" * 80)

skriptpfad = Path(__file__).parent
csvdatei_train = skriptpfad / 'data' / 'train' / 'ex_1.csv'

daten_train = pd.read_csv(csvdatei_train)
eingangssignal_train = daten_train['input_voltage'].values
ausgangssignal_train = daten_train['el_power'].values

# Normalizacijski parametri
minEingabe = eingangssignal_train.min()
maxEingabe = eingangssignal_train.max()
minAusgabe = ausgangssignal_train.min()
maxAusgabe = ausgangssignal_train.max()

print(f"✅ Normalizacijski parametri učitani")
print(f"   Ulaz: min={minEingabe:.2f}V, max={maxEingabe:.2f}V")
print(f"   Izlaz: min={minAusgabe:.2f}W, max={maxAusgabe:.2f}W")

# ============================================================================
# KORAK 4: GENERIRAJ SLIKE
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 3: GENERIRANJE SLIKA")
print("=" * 80)

N = 451

test_datoteke = {
    'ex_4.csv': skriptpfad / 'data' / 'test' / 'ex_4.csv',
    'ex_22.csv': skriptpfad / 'data' / 'test' / 'ex_22.csv'
}

# Za svaki model
for model_naziv, modell in ucitani_modeli.items():
    print(f"\n📊 Generiram slike za: {model_naziv}")

    # Za svaki dataset
    for naziv_dataseta, putanja_dataseta in test_datoteke.items():
        if not putanja_dataseta.exists():
            print(f"   ❌ {naziv_dataseta} nije pronađena")
            continue

        print(f"   📈 Obrađujem: {naziv_dataseta}")

        # Učitaj datoteku
        daten_test = pd.read_csv(putanja_dataseta)
        ulaz_test = daten_test['input_voltage'].values
        izlaz_test = daten_test['el_power'].values

        # Kreiraj sekvence
        ulazSekvence = []
        izlazSekvence = []

        for i in range(len(ulaz_test) - N):
            ulazSekvence.append(ulaz_test[i:i+N])
            izlazSekvence.append(izlaz_test[i+N])

        ulazSekvence = np.array(ulazSekvence)
        izlazSekvence = np.array(izlazSekvence)

        # Normalizacija
        ulazSekvence_norm = (ulazSekvence - minEingabe) / (maxEingabe - minEingabe)
        izlazSekvence_norm = (izlazSekvence - minAusgabe) / (maxAusgabe - minAusgabe)

        # Predviđanja
        ulazTensor = torch.FloatTensor(ulazSekvence_norm).unsqueeze(-1).to(gerät)
        izlazTensor = torch.FloatTensor(izlazSekvence_norm).unsqueeze(-1).to(gerät)

        with torch.no_grad():
            predviđanja = modell(ulazTensor)

        # Denormalizacija
        predviđanja_denorm = predviđanja.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe
        izlaz_denorm = izlazTensor.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe

        # Izračunaj RMSE
        RMSE = np.sqrt(np.mean((predviđanja_denorm - izlaz_denorm) ** 2))

        # ============================================================================
        # KREIRAJ SLIKU
        # ============================================================================

        fig, ax = plt.subplots(figsize=(14, 6))

        x_os = np.arange(len(izlaz_denorm))

        # Plot stvarnih vrijednosti
        ax.plot(x_os, izlaz_denorm.flatten(), 'b-', linewidth=2, label='Stvarna vrijednost', alpha=0.8)

        # Plot prognoza
        ax.plot(x_os, predviđanja_denorm.flatten(), 'r--', linewidth=2, label='Prognoza', alpha=0.7)

        # Formatiranje
        ax.set_xlabel('Vrijeme (sekvence)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Električna Snaga (W)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_naziv}\n{naziv_dataseta} - Prognoza vs Stvarno\nRMSE = {RMSE:.2f} W',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Spremi sliku
        slika_ime = f"plot_{model_naziv.replace('.pth', '')}_{naziv_dataseta.replace('.csv', '')}.png"
        plt.savefig(slika_ime, dpi=150, bbox_inches='tight')
        print(f"      ✅ Slika spremeljena: {slika_ime}")

        plt.close(fig)

print("\n" + "=" * 80)
print("SLIKE GENERIRANE!")
print("=" * 80)
print("\n✅ Sve slike su spremeljena u trenutnom direktoriju:")
print("   - plot_einfachesRNN_ex_4.png")
print("   - plot_einfachesRNN_ex_22.png")
print("   - plot_einfachesRNN1_ex_4.png")
print("   - plot_einfachesRNN1_ex_22.png")
print("\n🎉 Možeš ih preuzeti i pregledati!")
