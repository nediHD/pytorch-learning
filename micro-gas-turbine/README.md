# Gas Turbine RNN - Projektdokumentation

## 📋 Projekt Übersicht

Dieses Projekt implementiert ein einfaches **RNN mit LSTM** zur Vorhersage der elektrischen Leistung einer Mikrogasturbine basierend auf Eingangsspannungssignalen.

---

## 🎯 Was Wir Getan Haben

### Phase 1: Einfaches RNN (Ohne Knowledge-Guided Learning)

#### 1.1 Modellarchitektur
```
Input (N=451 Zeitschritte)
    ↓
LSTM Schicht 1 (32 Neuronen)
    ↓
LSTM Schicht 2 (32 Neuronen)
    ↓
LSTM Schicht 3 (32 Neuronen)
    ↓
Dense Schicht (1 Ausgang)
    ↓
Output (Leistungsprognose)
```

**Parameter:**
- Eingabegröße: N = 451 (Fenster für vergangene 451 Sekunden)
- LSTM Schichten: 3
- Hidden Units pro Schicht: 32
- Aktivierungsfunktion: LSTM-Standard (Sigmoid + Tanh)
- Output: 1 Wert (Leistung in W)

#### 1.2 Trainingsdaten
**Quelle:** `data/train/ex_1.csv`
- Gesamtdaten: 9920 Zeilen
- Zeitserie: 810s - 10720s (~2.75 Stunden)
- Eingangssignal: `input_voltage` (3-10V)
- Ausgangssignal: `el_power` (932.84-3249.89W)

**Sekvenzenvorbereitung:**
- Sliding Window Ansatz mit N=451
- Erstellt: 9469 Trainingssequenzen (9920 - 451)
- Normalisierung: Min-Max Skalierung auf [0,1]

#### 1.3 Trainingsparameter
```
Lernrate (Learning Rate):     0.001
Batch Size:                   32
Optimizer:                    Adam
Verlustfunktion:              MSE (Mean Squared Error)
Epochen:                      300
Batches pro Epoche:           296 (9469 / 32)
Gesamte Gewichtsupdates:      88.800 (300 × 296)
GPU Support:                  CUDA (falls verfügbar)
Early Stopping:               Speichert bestes Modell
```

#### 1.4 Trainingsprozess
```
KORAK 1: Daten laden und Sekvenzen erstellen
KORAK 2: Daten normalisieren (0-1 Bereich)
KORAK 3: PyTorch DataLoader mit Batch-Verarbeitung
KORAK 4: RNN Modell definieren (3 LSTM + Dense)
KORAK 5: MSE Loss Funktion konfigurieren
KORAK 6: 300 Epochen trainieren
   └─ Für jede Epoche:
      ├─ 296 Batches verarbeiten
      ├─ Forward Pass durch LSTM
      ├─ MSE Verlust berechnen
      ├─ Backward Pass (Backpropagation Through Time)
      ├─ Gewichte mit Adam aktualisieren
      └─ Durchschnittlichen Verlust speichern
   └─ Bestes Modell speichern (kleinster Verlust)
KORAK 7: Trainiertes Modell herunterladen
```

---

## 📊 Trainingsergebnisse

### Verlauf (Aus Google Colab)
```
Epoche 1:   MSE = 0.924282
Epoche 2:   MSE = 0.856471
...
Epoche 26:  MSE = 0.008252
Epoche 37:  MSE = 0.008252 ← ~20 Sekunden pro Epoche
...
Epoche 300: MSE = 0.006104
Beste MSE:  0.006104
```

**Trainingszeit:** ~2 Stunden auf Google Colab GPU (T4)

---

## 🧪 Test-Ergebnisse

### Test auf ex_4.csv ✅
```
Dateiname:       ex_4.csv
Redaka:          9.795 Zeilen
Testsäquenzen:   9.344
Zeitraum:        ~2.7 Stunden

RMSE:  384.19 W    ← Durchschnittlicher Fehler
MAE:   333.14 W    ← Mittlerer Absoluter Fehler
MSE:   0.027493    ← Normalisierter Fehler

BEWERTUNG: ✅ GUT (~11% Fehler)
```

### Test auf ex_22.csv ❌
```
Dateiname:       ex_22.csv
Redaka:          8.490 Zeilen
Testsequenzen:   8.039
Zeitraum:        ~2.4 Stunden

RMSE:  688.63 W    ← Durchschnittlicher Fehler
MAE:   532.05 W    ← Mittlerer Absoluter Fehler
MSE:   0.088328    ← Normalisierter Fehler

BEWERTUNG: ❌ SCHLECHT (~22% Fehler, 1.8x schlechter als ex_4)
```

### Vergleich
| Datei   | RMSE    | MAE    | Bewertung |
|---------|---------|--------|-----------|
| ex_4    | 384 W   | 333 W  | ✅ Gut    |
| ex_22   | 689 W   | 532 W  | ❌ Schlecht |

**Grund für Unterschied:** ex_22 enthält schnellere Übergänge, die das Modell schwer vorhersagt.

---

## 📁 Projektstruktur

```
pytorch-learning/
├── micro-gas-turbine/
│   ├── einfacheRNN_Colab.py          ← Hauptcode (Google Colab)
│   ├── test_model.py                 ← Test-Script
│   ├── einfachesRNN.pth              ← Trainiertes Modell (lokal)
│   ├── README.md                     ← Diese Datei
│   ├── data/
│   │   ├── train/
│   │   │   └── ex_1.csv              ← Trainingsdaten
│   │   ├── ex_4.csv                  ← Testdaten
│   │   └── ex_22.csv                 ← Testdaten
│   └── Knowledge-Guided Learning...pdf ← Wissenschaftliches Papier
└── .git/                             ← Git Repository
```

---

## 🚀 Wie Man Das Projekt Nutzt

### 1. Google Colab (Empfohlen)
```
1. Öffne https://colab.research.google.com/
2. Aktiviere GPU: Settings → Accelerator → GPU
3. Kopiere Code aus einfacheRNN_Colab.py
4. Zalijepi in Colab Cell
5. Führe `files.upload()` aus
6. Starte Training mit Shift+Enter
```

### 2. Lokales Training (Falls GPU vorhanden)
```bash
cd micro-gas-turbine
python einfacheRNN.py
```

### 3. Modell Testen
```bash
cd micro-gas-turbine
python test_model.py
```

---

## 📈 Nächste Schritte (Knowledge-Guided Learning)

Das Originalpapier schlägt vor:
- **Permissible System States:** Nur 3 erlaubte Leistungsänderungsraten:
  - Rising: +6.388 W/s
  - Falling: -6.388 W/s
  - Stationary: 0 W/s

- **Multi-State Constraint:** Penalty in Loss Funktion für Abweichungen
- **Erwartete Verbesserung:** 40% Reduktion des Vorhersagefehlers

---

## 📚 Referenzen

**Papier:** Knowledge-Guided Learning of Temporal Dynamics and its Application to Gas Turbines
- Autoren: Pawel Bielski, Aleksandr Eismont, Jakob Bach, Florian Leiser, Dustin Kottonau, Klemens Böhm
- Konferenz: E-Energy '24, Juni 2024, Singapur
- DOI: 10.1145/3632775.3661967

---

## 🛠️ Technologie Stack

- **Deep Learning Framework:** PyTorch 2.x
- **Datenverarbeitung:** Pandas, NumPy
- **GPU Support:** CUDA (NVIDIA)
- **Umgebung:** Google Colab, Jupyter Notebook
- **Version Control:** Git / GitHub
- **Programmiersprache:** Python 3.8+

---

## 📝 Notizen

### Was Funktioniert Gut:
- ✅ RNN mit LSTM kann zeitliche Abhängigkeiten lernen
- ✅ Auf ex_4.csv: ~11% durchschnittlicher Fehler
- ✅ Google Colab GPU macht Training praktikabel (2 Stunden)
- ✅ Modell speichern/laden funktioniert

### Was Problematisch Ist:
- ❌ ex_22.csv: ~22% Fehler (schnelle Übergänge schwer vorherzusagen)
- ❌ Ohne Knowledge-Guided Learning: Fehlerquoten suboptimal
- ❌ Modell generalisiert unterschiedlich auf verschiedene Datensätze

---

**Zuletzt aktualisiert:** 2026-06-30
**Status:** Einfaches RNN abgeschlossen, Knowledge-Guided Learning in Planung
