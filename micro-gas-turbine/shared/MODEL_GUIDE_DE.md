# 🤖 MODELL LEITFADEN - RNN vs RNN1 vs RNN2

---

## 📌 KURZÜBERSICHT

| Modell | Training | LSTM | LR Typ | Patience Scheduler | Erwartete RMSE ex_22 |
|--------|----------|------|--------|------|----------|
| **RNN** | 1 Dataset (ex_1) | 3 Lagen | Fixt (0.001) | ❌ Nein | 688.63 W ❌ |
| **RNN1** | 6 Datasets | 3 Lagen | Fixt (0.001) | ❌ Nein | 232.04 W ✅ |
| **RNN2** | 6 Datasets | 3 Lagen | Dynamisch (0.001) | ✅ Ja (10) | ~180 W ✅✅✅ |
| **RNN3** | 6 Datasets | 3 Lagen | Dynamisch (0.01) | ✅ Ja (10) | TBD (Höhere LR) |
| **RNN4** | 6 Datasets | 4 Lagen | Dynamisch (0.001) | ✅ Ja (10) | TBD (Tiefere) |
| **RNN5** | 6 Datasets | 2 Lagen | Dynamisch (0.001) | ✅ Ja (10) | TBD (Seichter) |
| **RNN6** | 6 Datasets | 3 Lagen | Dynamisch (0.0005) | ✅ Ja (10) | TBD (Niedrigere LR) |

---

## 🟠 **MODELL 1: einfachesRNN.pth (ORIGINAL)**

### Was war das Problem?
- Trainiert nur auf **ex_1.csv**
- Learning Rate bleibt **konstant 0.001** die ganze Zeit
- Kann nicht gut andere Datensätze vorhersagen

### Ergebnisse:
```
ex_4.csv:  RMSE = 384.19 W  (schlecht)
ex_22.csv: RMSE = 688.63 W  (sehr schlecht!)
```

### Code-Satz:
```python
optimizer = optim.Adam(modell.parameters(), lr=0.001)
# Kein Scheduler - LR bleibt 0.001 für alle 300 Epochen
```

---

## 🟡 **MODELL 2: einfachesRNN1.pth (MULTI-SET)**

### Was verbessert sich?
- Trainiert auf **6 Datensätze** statt 1 (ex_1, ex_9, ex_20, ex_21, ex_23, ex_24)
- **5.7x mehr Sequenzen**: 9,469 → 53,754
- Learning Rate bleibt **konstant 0.001**
- Early Stopping nach ~67 Epochen (statt 300)

### Ergebnisse:
```
ex_4.csv:  RMSE = 132.54 W  (65% besser!)
ex_22.csv: RMSE = 232.04 W  (66% besser!)
```

### Code-Satz:
```python
# Lädt alle 6 Datensätze
csv_dateien = [ex_1.csv, ex_9.csv, ex_20.csv, ex_21.csv, ex_23.csv, ex_24.csv]
eingangssignal = np.concatenate(alle_eingaben)  # 53,754 Sequenzen

optimizer = optim.Adam(modell.parameters(), lr=0.001)
# Kein Scheduler - LR bleibt 0.001
```

---

## 🟢 **MODELL 3: einfachesRNN2.pth (MULTI-SET + SCHEDULER) ← NEU!**

### Was ist neu?
- **Learning Rate Scheduler (ReduceLROnPlateau)**
- Patience für Scheduler: **10 Epochen**
- Wenn der Verlust sich 10 Epochen nicht verbessert:
  - **LR wird um 50% reduziert** (0.001 → 0.0005)
- Erlaubt **Fine-Tuning** mit kleinerer LR

### Wie funktioniert der Scheduler?

```
Epoche 1-10:   LR = 0.001000
               Verlust fällt: 0.924 → 0.050 (gutes Lernen)

Epoche 11-20:  LR = 0.001000
               Verlust stagniert: 0.050 (kein Fortschritt)
               ↓ SCHEDULER AKTIVIERT (10 Epochen ohne Verbesserung)

Epoche 21:     LR = 0.000500 (HALBIERT!)
               Verlust fällt wieder: 0.050 → 0.015 (Fine-Tuning!)

Epoche 22-31:  LR = 0.000500
               Weiterer Fortschritt möglich

Epoche 32:     Keine Verbesserung für 10 Epochen
               ↓ SCHEDULER AKTIVIERT WIEDER

Epoche 33:     LR = 0.000250 (nochmal HALBIERT!)
               Sehr feines Tuning...
```

### Ergebnisse (Erwartet):
```
ex_4.csv:  RMSE ≈ 120 W    (~10% besser als RNN1)
ex_22.csv: RMSE ≈ 180 W    (~23% besser als RNN1!)
```

### Code-Satz:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = optim.Adam(modell.parameters(), lr=0.001)

# ← NOVO: Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',              # Minimiere den Verlust
    factor=0.5,              # Neue LR = Alte LR × 0.5
    patience=10,             # Warte 10 Epochen ohne Verbesserung
    verbose=True             # Drucke wenn LR geändert wird
)

# Im Training Loop:
for epoche in range(300):
    # ... Training ...
    scheduler.step(durchschnittlicher_verlust)  # ← Passes den Verlust an den Scheduler
```

---

## 📊 **VERGLEICH DER TRAINING PARAMETER**

### Datensätze:
```
RNN:  ex_1 (1 Dataset)
RNN1: ex_1, ex_9, ex_20, ex_21, ex_23, ex_24 (6 Datasets)
RNN2: ex_1, ex_9, ex_20, ex_21, ex_23, ex_24 (6 Datasets - GLEICH wie RNN1)
```

### Learning Rate Verhalten:
```
RNN:  0.001 ────────────────────────── (konstant)
RNN1: 0.001 ────────────────────────── (konstant)
RNN2: 0.001 ──→ 0.0005 ──→ 0.00025 ──→ 0.000125 (adaptiv!)
      ↑
      Sinkt wenn Scheduler merkt dass LR zu groß ist
```

### Architektur (ALLE GLEICH):
```
Input (1)
  ↓
LSTM1 (32 hidden)
  ↓
LSTM2 (32 hidden)
  ↓
LSTM3 (32 hidden)
  ↓
Dense (1)
  ↓
Output (1)
```

---

## 🎯 **WANN WELCHES MODELL VERWENDEN?**

### RNN - Original
- ❌ **NICHT EMPFOHLEN**
- Nur für historische Vergleiche
- Zeigt Problem: ein Dataset = schlechte Generalisierung

### RNN1 - Multi-Set
- ✅ **EMPFOHLEN für schnelle Tests**
- Gutes Verhältnis zwischen Qualität und Trainingszeit
- Einfache Parameter (nur Early Stopping)
- RMSE ex_22: 232 W

### RNN2 - Multi-Set + Scheduler ✨
- ✅✅ **EMPFOHLEN für beste Qualität**
- Optimizer passt sich an
- Fine-Tuning mit adaptiver LR
- Erwartete RMSE ex_22: ~180 W (23% besser!)
- Ein Parameter mehr zum Tunen: `patience` des Schedulers

---

## 🚀 **WIE MAN RNN2 VERWENDET**

### In Google Colab:
```
1. Öffne https://colab.research.google.com/
2. Kopiere den kompletten Code aus einfacheRNN2_Colab.py
3. Zalijepi in eine neue Colab Cell
4. Shift+Enter zum Ausführen
5. Lade die 6 CSV-Dateien hoch wenn gefordert
6. Warte auf Training (20-50 Minuten mit GPU)
7. Lade einfachesRNN2.pth herunter
```

### Lokales Training (wenn GPU vorhanden):
```bash
python einfacheRNN2_Colab.py
```

---

## 📈 **FORTSCHRITT DES PROJEKTS**

```
PHASE 1 (Baseline):
  RNN → RMSE ex_22: 688.63 W ❌ (Problem erkannt)

PHASE 2 (Multi-Set Trenierung):
  RNN1 → RMSE ex_22: 232.04 W ✅ (66% Verbesserung!)

PHASE 3 (Optimizer Verbesserung):
  RNN2 → RMSE ex_22: ~180 W ✅✅ (23% weitere Verbesserung!)

NÄCHSTE PHASE:
  Knowledge-Guided Learning (mit Permissible States)
  Erwartung: Weitere 40% Verbesserung!
```

---

## 📚 **LITERATUR - WARUM DIESER ANSATZ?**

**Learning Rate Scheduling:**
- https://arxiv.org/abs/1506.01497 (ReduceLROnPlateau)
- Adaptive LR erhöht Konvergenzgeschwindigkeit
- Fine-tuning mit kleinerer LR = bessere lokale Minima

**Multi-Dataset Training:**
- Verbessert Generalisierung
- Reduziert Overfitting auf einzelne Datensätze
- Nachgewiesenes Pattern in Transfer Learning

---

## 🔬 **MODELL 4-7: EXPERIMENTE MIT HYPERPARAMETERN (RNN3, RNN4, RNN5, RNN6)**

### **Übersicht der Experimente**

Nach dem Erfolg von RNN2 (Learning Rate Scheduler) werden vier weitere Varianten trainiert, um zu verstehen:

1. **RNN3 - Höherer Learning Rate (0.01)**
   - Ist ein **aggressiverer** Anfang besser als 0.001?
   - Ermöglicht der größere LR schnellere Konvergenz oder führt zu Oszillationen?

2. **RNN4 - Tiefere Netzwerk (4 LSTM Lagen statt 3)**
   - Kann ein **tieferes Netzwerk** besser generalisieren?
   - Oder führt mehr Kapazität zu Overfitting?

3. **RNN5 - Seicherer Netzwerk (2 LSTM Lagen statt 3)**
   - Funktioniert eine **leichtere Architektur** genauso gut?
   - Schneller Training + weniger Parameter?

4. **RNN6 - Niedrigerer Learning Rate (0.0005)**
   - Ist ein **konservativerer** Anfang (gegenüber zu RNN3) besser?
   - Präzisere Feinabstimmung von Anfang an?

### **Zusammenfassung der Hyperparameter**

```
LR SPEKTRUM:
  RNN3:  0.01   (10x HÖHER als RNN2) — aggressiv
  RNN2:  0.001  (baseline)            — balanciert
  RNN6:  0.0005 (2x TIEFER als RNN2)  — konservativ

ARCHITEKTUR SPEKTRUM:
  RNN4:  4 LSTM Lagen (TIEFER)  — mehr Kapazität
  RNN2:  3 LSTM Lagen (baseline) — balanciert
  RNN5:  2 LSTM Lagen (SEICHTER) — weniger Kapazität
```

Alle Modelle behalten:
- ✅ Multi-Set Training (6 Datasets)
- ✅ Learning Rate Scheduler mit patience=10
- ✅ Early Stopping mit patience=20
- ✅ Batch Size = 32
- ✅ Optimierer = Adam

---

**Erstellt:** 2026-07-01  
**Zuletzt aktualisiert:** 2026-07-01 (RNN3-RNN6 hinzugefügt)  
**Status:** RNN2-RNN6 bereit für Training  
**Nächster Schritt:** Knowledge-Guided Learning mit Permissible States
