# 🔬 MODELVERGLEICH - RNN bis RNN6 Experimente

---

## 📋 ÜBERBLICK

Dieses Dokument dokumentiert die schrittweise Verbesserung der RNN-Modelle für die Vorhersage der Gastrubinen-Leistung.

### Evolutionspfad:

```
RNN (Baseline: 1 Dataset)
   ↓
RNN1 (Multi-Set: 6 Datasets)
   ↓
RNN2 (Learning Rate Scheduler)
   ↙        ↙         ↘       ↘
RNN3    RNN4       RNN5      RNN6
(Höher) (Tiefer) (Seichter) (Niedriger)
```

---

## 🟠 MODELL 1: einfachesRNN.pth (BASELINE)

**Datei:** `einfacheRNN_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 1 (ex_1.csv) |
| Anzahl Sequenzen | 9,469 |
| LSTM Lagen | 3 |
| Learning Rate | 0.001 (konstant) |
| Scheduler | ❌ Nein |
| Early Stopping | ✅ Ja (Patience=20) |
| Erwartete RMSE ex_22 | 688.63 W ❌ |

### Problem:
- Trainiert auf nur 1 Datensatz
- Generalisiert schlecht auf andere Datensätze
- Zeigt Overfitting-Problem

---

## 🟡 MODELL 2: einfachesRNN1.pth (MULTI-SET)

**Datei:** `einfacheRNN_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (ex_1, ex_9, ex_20, ex_21, ex_23, ex_24) |
| Anzahl Sequenzen | ~53,754 (5.7x mehr!) |
| LSTM Lagen | 3 |
| Learning Rate | 0.001 (konstant) |
| Scheduler | ❌ Nein |
| Early Stopping | ✅ Ja (Patience=20) |
| Erwartete RMSE ex_22 | 232.04 W ✅ |

### Verbesserung:
- **66% bessere RMSE** gegen RNN!
- Multi-Dataset Training hilft enorm
- Aber: Learning Rate bleibt konstant

### Lernpunkt:
**Datenvariabilität ist entscheidend!**

---

## 🟢 MODELL 3: einfachesRNN2.pth (SCHEDULER)

**Datei:** `einfacheRNN2_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (GLEICH wie RNN1) |
| Anzahl Sequenzen | ~53,754 (GLEICH wie RNN1) |
| LSTM Lagen | 3 |
| Learning Rate (Start) | 0.001 |
| LR Scheduler | ✅ ReduceLROnPlateau |
| Scheduler Patience | 10 Epochen |
| Scheduler Factor | 0.5 (halbiert) |
| Early Stopping | ✅ Ja (Patience=20) |
| Erwartete RMSE ex_22 | ~180 W ✅✅ |

### Verbesserung:
- **23% bessere RMSE** gegen RNN1!
- Scheduler aktiviert bei Stagnation:
  - Epoch 1-10: LR = 0.001
  - Epoch 21: LR = 0.0005 (aktiviert!)
  - Epoch 33: LR = 0.00025 (aktiviert wieder!)

### Lernpunkt:
**Adaptive Learning Rates ermöglichen feineres Tuning!**

---

## 🔵 MODELL 4: einfachesRNN3.pth (HÖHERE LR - EXPERIMENT 1)

**Datei:** `einfacheRNN3_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (GLEICH wie RNN2) |
| Anzahl Sequenzen | ~53,754 (GLEICH wie RNN2) |
| LSTM Lagen | 3 (GLEICH wie RNN2) |
| **Learning Rate (Start)** | **0.01** (10x HÖHER!) |
| LR Scheduler | ✅ ReduceLROnPlateau (GLEICH) |
| Scheduler Patience | 10 Epochen |
| Early Stopping | ✅ Ja (Patience=20) |
| RMSE ex_22 | TBD (nach Trenierung) |

### Experiment:
**Fragestellung:** Ermöglicht ein aggressiverer Anfang schnellere Konvergenz?

### Hypothesen:
- ✅ **Optimistisch:** Schnelleres Lernen am Anfang + Scheduler-Anpassung = besser
- ❌ **Pessimistisch:** Zu hohe LR kann zu Oszillationen führen = schlechter

### LR Progression (erwartet):
```
Epoch 1:     LR = 0.01
Epoch 1-10:  Schnelle Konvergenz
Epoch 20:    Stagnation (bei viel höherer LR)
Epoch 21:    LR = 0.005 (Scheduler aktiviert)
Epoch 31:    Weitere Stagnation?
Epoch 32:    LR = 0.0025 (Scheduler aktiviert wieder)
```

---

## 🟣 MODELL 5: einfachesRNN4.pth (TIEFERES NETZWERK - EXPERIMENT 2)

**Datei:** `einfacheRNN4_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (GLEICH wie RNN2) |
| Anzahl Sequenzen | ~53,754 (GLEICH wie RNN2) |
| **LSTM Lagen** | **4** (statt 3 - +1 Schicht!) |
| LSTM Konfiguration | 1→32→32→32→32→1 |
| Learning Rate (Start) | 0.001 (GLEICH wie RNN2) |
| LR Scheduler | ✅ ReduceLROnPlateau (GLEICH) |
| Scheduler Patience | 10 Epochen |
| Early Stopping | ✅ Ja (Patience=20) |
| RMSE ex_22 | TBD (nach Trenierung) |

### Experiment:
**Fragestellung:** Kann ein tieferes Netzwerk komplexere Muster lernen?

### Hypothesen:
- ✅ **Optimistisch:** Mehr Lagen = mehr Kapazität = bessere Generalisierung
- ❌ **Pessimistisch:** Zu viele Parameter = Overfitting auf Training Data

### Architekturvergleich:
```
RNN2:
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → Dense(1) → Output
  
RNN4:
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → LSTM4(32) → Dense(1) → Output
                                                    ↑↑↑↑ NEUE SCHICHT!
```

### Komplexitätsvergleich:
- RNN2: ~3,105 Parameter
- RNN4: ~4,145 Parameter (+33% mehr!)

---

## 🟢 MODELL 6: einfachesRNN5.pth (SEICHERES NETZWERK - EXPERIMENT 3)

**Datei:** `einfacheRNN5_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (GLEICH wie RNN2) |
| Anzahl Sequenzen | ~53,754 (GLEICH wie RNN2) |
| **LSTM Lagen** | **2** (statt 3 - -1 Schicht!) |
| LSTM Konfiguration | 1→32→32→1 |
| Learning Rate (Start) | 0.001 (GLEICH wie RNN2) |
| LR Scheduler | ✅ ReduceLROnPlateau (GLEICH) |
| Scheduler Patience | 10 Epochen |
| Early Stopping | ✅ Ja (Patience=20) |
| RMSE ex_22 | TBD (nach Trenierung) |

### Experiment:
**Fragestellung:** Brauchen wir wirklich 3 LSTM-Lagen?

### Hypothesen:
- ✅ **Optimistisch:** Einfacheres Modell = schneller Training + weniger Overfitting
- ❌ **Pessimistisch:** Zu wenig Kapazität = kann nicht alle Muster lernen

### Architekturvergleich:
```
RNN2:
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → Dense(1) → Output
  
RNN5:
  Input(1) → LSTM1(32) → LSTM2(32) → Dense(1) → Output
                         ↑↑↑↑ DIREKT ZUM OUTPUT (keine LSTM3)
```

### Komplexitätsvergleich:
- RNN2: ~3,105 Parameter
- RNN5: ~2,065 Parameter (-33% weniger!)
- **Vorteil:** Schnelleres Training, weniger Speicher

---

## 🟠 MODELL 7: einfachesRNN6.pth (NIEDRIGERE LR - EXPERIMENT 4)

**Datei:** `einfacheRNN6_Colab.py`

| Eigenschaft | Wert |
|------------|------|
| Trainingsdatensätze | 6 (GLEICH wie RNN2) |
| Anzahl Sequenzen | ~53,754 (GLEICH wie RNN2) |
| LSTM Lagen | 3 (GLEICH wie RNN2) |
| **Learning Rate (Start)** | **0.0005** (2x NIEDRIGER!) |
| LR Scheduler | ✅ ReduceLROnPlateau (GLEICH) |
| Scheduler Patience | 10 Epochen |
| Early Stopping | ✅ Ja (Patience=20) |
| RMSE ex_22 | TBD (nach Trenierung) |

### Experiment:
**Fragestellung:** Ist ein konservativerer Anfang besser als aggressiv?

### Hypothesen:
- ✅ **Optimistisch:** Präziserer Start + Scheduler = beste Konvergenz
- ❌ **Pessimistisch:** Zu langsam am Anfang = verspätete Konvergenz

### LR Spektrum der RNN3/RNN2/RNN6 Vergleiche:
```
RNN3: 0.01   ← AGGRESSIV (schnell am Anfang)
RNN2: 0.001  ← BALANCED (Standard)
RNN6: 0.0005 ← KONSERVATIV (präzise am Anfang)
```

### LR Progression (erwartet):
```
Epoch 1:     LR = 0.0005
Epoch 1-10:  Langsame, aber präzise Konvergenz
Epoch 20:    Stagnation bei niedriger LR?
Epoch 21:    LR = 0.00025 (Scheduler aktiviert)
Epoch 30:    Sehr feines Tuning...
```

---

## 📊 VERGLEICHSTABELLE - ALLE 7 MODELLE

| Aspekt | RNN | RNN1 | RNN2 | RNN3 | RNN4 | RNN5 | RNN6 |
|--------|-----|------|------|------|------|------|------|
| **Trainingsdatensätze** | 1 | 6 | 6 | 6 | 6 | 6 | 6 |
| **Sequenzen** | 9.5k | 53.7k | 53.7k | 53.7k | 53.7k | 53.7k | 53.7k |
| **LSTM Lagen** | 3 | 3 | 3 | 3 | **4** | **2** | 3 |
| **Parameter** | ~3105 | ~3105 | ~3105 | ~3105 | ~4145 | ~2065 | ~3105 |
| **LR (Start)** | 0.001 | 0.001 | 0.001 | **0.01** | 0.001 | 0.001 | **0.0005** |
| **Scheduler** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Scheduler Patience** | - | - | 10 | 10 | 10 | 10 | 10 |
| **Early Stop Patience** | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| **Experiment** | Baseline | Multi-Set | Baseline LR | Höhere LR | Tieferes Netz | Seicheres Netz | Niedrigere LR |
| **Erwartete RMSE ex_22** | 688 W ❌ | 232 W ✅ | ~180 W ✅✅ | TBD | TBD | TBD | TBD |

---

## 🎯 EXPERIMENTALDESIGN

### Zielsetzungen:

1. **Verstehen Sie LR Impact:** RNN3 vs RNN6 zeigen Grenzen des Learning Rate Spektrums
2. **Verstehen Sie Architektur:** RNN4 vs RNN5 zeigen Impact der Netzwerktiefe
3. **Baseline:** RNN2 ist der Vergleichsmaßstab (balanced configuration)

### Erwartete Erkenntnisse:

Nach dem Training aller Modelle sollte man verstehen:

```
Wenn RNN3 besser als RNN2:
  → Höhere LR ist vorteilhaft für dieses Problem
  → Scheduler bietet genug Schutz vor Oszillationen
  
Wenn RNN3 schlechter als RNN2:
  → 0.001 ist bereits optimal
  → Zu hohe LR schadet trotz Scheduler
  
Wenn RNN4 besser als RNN2:
  → Mehr Kapazität hilft
  → Kein Overfitting-Problem
  
Wenn RNN4 schlechter als RNN2:
  → Tiefere Netzwerke nicht notwendig
  → Möglicherweise Overfitting
  
Wenn RNN5 genauso gut wie RNN2:
  → 2 Lagen sind ausreichend
  → Schneller Training ohne Qualitätsverlust
  
Wenn RNN6 besser als RNN2:
  → Konservativere Anfangsphase hilft
  → Bessere Feinabstimmung von Anfang an
```

---

## 📁 DATEISTRUKTUR

```
micro-gas-turbine/
├── einfacheRNN_Colab.py        ← RNN (Original) und RNN1
├── einfacheRNN2_Colab.py       ← RNN2 (Scheduler)
├── einfacheRNN3_Colab.py       ← RNN3 (Höhere LR)
├── einfacheRNN4_Colab.py       ← RNN4 (4 LSTM Lagen)
├── einfacheRNN5_Colab.py       ← RNN5 (2 LSTM Lagen)
├── einfacheRNN6_Colab.py       ← RNN6 (Niedrigere LR)
├── test_model.py               ← Test alle Modelle (aktualisiert)
├── plot_modeli.py              ← Visualisierung (aktualisiert)
├── TRAINING_LOG.md             ← Detaillierte Hyperparameter
├── MODEL_GUIDE_DE.md           ← Modellvorhersagen
└── MODELS_COMPARISON_DE.md     ← Dieser File
```

---

## 🚀 NÄCHSTE SCHRITTE

1. **Google Colab:** Lade jede Skriptdatei (RNN3-RNN6) in Colab hoch
2. **Training:** Trainiere jeden Modell mit GPU
3. **Download:** Speichere einfachesRNN3.pth ... einfachesRNN6.pth
4. **Testing:** Nutze test_model.py um alle Modelle zu vergleichen
5. **Analyse:** Nutze plot_modeli.py um Vorhersagen zu visualisieren
6. **Documentation:** Update RMSE Werte in TRAINING_LOG.md nach Training

---

**Erstellt:** 2026-07-01  
**Status:** Bereit für Colab-Training  
**Nächste Phase:** Knowledge-Guided Learning mit Permissible States

