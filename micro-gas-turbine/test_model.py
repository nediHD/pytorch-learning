# -*- coding: utf-8 -*-
# ============================================================================
# USPOREDBA DVA MODELA - EVALUACIJA NA ex_4.csv I ex_22.csv
# ============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("USPOREDBA DVAJU MODELA: einfachesRNN.pth vs einfachesRNN1.pth")
print("TEST NA: ex_4.csv I ex_22.csv")
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
    'einfachesRNN.pth': 'Original (ex_1 samo)',
    'einfachesRNN1.pth': 'Multi-Set + Early Stopping'
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
# KORAK 3: UČITAJ TRAINING PODATKE (za normalizacijske parametre)
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
# KORAK 4: TESTIRAJ SVE MODELE NA SVE DATASETE
# ============================================================================

print("\n" + "=" * 80)
print("KORAK 3: TESTIRANJE MODELA")
print("=" * 80)

N = 451

test_datoteke = {
    'ex_4.csv': skriptpfad / 'data' / 'test' / 'ex_4.csv',
    'ex_22.csv': skriptpfad / 'data' / 'test' / 'ex_22.csv'
}

# Struktura: rezultati[model_naziv][dataset_naziv] = {MSE, RMSE, MAE, ...}
rezultati = {}

for model_naziv, modell in ucitani_modeli.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_naziv}")
    print(f"{'='*80}")

    rezultati[model_naziv] = {}

    for naziv_dataseta, putanja_dataseta in test_datoteke.items():
        if not putanja_dataseta.exists():
            print(f"❌ {naziv_dataseta} nije pronađena na {putanja_dataseta}")
            continue

        print(f"\n📂 Testiram: {naziv_dataseta}")

        # Učitaj datoteku
        daten_test = pd.read_csv(putanja_dataseta)
        ulaz_test = daten_test['input_voltage'].values
        izlaz_test = daten_test['el_power'].values

        print(f"   Redaka: {len(daten_test)}")
        print(f"   Ulaz: min={ulaz_test.min():.2f}V, max={ulaz_test.max():.2f}V")
        print(f"   Izlaz: min={izlaz_test.min():.2f}W, max={izlaz_test.max():.2f}W")

        # Kreiraj sekvence
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

        # Predviđanja
        ulazTensor = torch.FloatTensor(ulazSekvence).unsqueeze(-1).to(gerät)
        izlazTensor = torch.FloatTensor(izlazSekvence).unsqueeze(-1).to(gerät)

        with torch.no_grad():
            predviđanja = modell(ulazTensor)
            loss = nn.MSELoss()(predviđanja, izlazTensor)

        # Denormalizacija
        predviđanja_denorm = predviđanja.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe
        izlaz_denorm = izlazTensor.cpu().numpy() * (maxAusgabe - minAusgabe) + minAusgabe

        # RMSE i MAE izračun
        RMSE = np.sqrt(np.mean((predviđanja_denorm - izlaz_denorm) ** 2))
        MAE = np.mean(np.abs(predviđanja_denorm - izlaz_denorm))

        rezultati[model_naziv][naziv_dataseta] = {
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
# KORAK 5: USPOREDBA MODELA
# ============================================================================

print("\n" + "=" * 80)
print("USPOREDBA MODELA - DETALJNA TABLICA")
print("=" * 80)

# Ispiši rezultate po datasetu
for naziv_dataseta in test_datoteke.keys():
    print(f"\n📊 DATASET: {naziv_dataseta}")
    print("-" * 80)

    print(f"{'Model':<30} {'RMSE (W)':<15} {'MSE':<15} {'MAE (W)':<15}")
    print("-" * 80)

    best_rmse = float('inf')
    best_model_rmse = None

    for model_naziv in rezultati.keys():
        if naziv_dataseta not in rezultati[model_naziv]:
            continue

        metrike = rezultati[model_naziv][naziv_dataseta]
        rmse = metrike['RMSE']
        mse = metrike['MSE']
        mae = metrike['MAE']

        # Označi najbolji model po RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_rmse = model_naziv

        print(f"{model_naziv:<30} {rmse:>14.2f} {mse:>14.6f} {mae:>14.2f}")

    if best_model_rmse:
        print("-" * 80)
        print(f"🏆 POBJEDNIK PO RMSE: {best_model_rmse}")

# ============================================================================
# SVEUKUPNA USPOREDBA
# ============================================================================

print("\n" + "=" * 80)
print("SVEUKUPNA USPOREDBA POBJEDNIKA PO DATASETU")
print("=" * 80)

najbolji_po_datasetu = {}

for naziv_dataseta in test_datoteke.keys():
    best_rmse = float('inf')
    best_model = None

    for model_naziv in rezultati.keys():
        if naziv_dataseta not in rezultati[model_naziv]:
            continue
        rmse = rezultati[model_naziv][naziv_dataseta]['RMSE']
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_naziv

    najbolji_po_datasetu[naziv_dataseta] = (best_model, best_rmse)

    print(f"\n{naziv_dataseta}:")
    print(f"  🏆 Najbolji model: {best_model}")
    print(f"  📊 RMSE: {best_rmse:.2f} W")

# ============================================================================
# DETALJNI ISPIS SVIH REZULTATA
# ============================================================================

print("\n" + "=" * 80)
print("DETALJNI REZULTATI PO MODELU I DATASETU")
print("=" * 80)

for model_naziv, opis in modeli_za_testiranje.items():
    if model_naziv not in rezultati:
        continue

    print(f"\n{'='*80}")
    print(f"Model: {model_naziv}")
    print(f"Opis: {opis}")
    print(f"{'='*80}")

    for naziv_dataseta in test_datoteke.keys():
        if naziv_dataseta not in rezultati[model_naziv]:
            continue

        metrike = rezultati[model_naziv][naziv_dataseta]

        print(f"\n  📂 {naziv_dataseta}:")
        print(f"     Sekvenci testirane: {metrike['sekvenci']}")
        print(f"     MSE (normalizirano): {metrike['MSE']:.6f}")
        print(f"     RMSE (W): {metrike['RMSE']:.2f}")
        print(f"     MAE (W): {metrike['MAE']:.2f}")

print("\n" + "=" * 80)
print("TEST ZAVRŠEN!")
print("=" * 80)
print("\n✅ Usporedba je kompletna. Vidiš koji je model bolji po svakom datasetu!")
print("🎉 Ako je einfachesRNN1.pth bolji, znači da multi-set treniranje sa Early Stopping-om")
print("   daje bolje rezultate!")
