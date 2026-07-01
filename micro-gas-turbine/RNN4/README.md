# 🟣 RNN4 (MULTI-SET + SCHEDULER + 4 LSTM SLOJA)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ISTO kao RNN2) |
| Broj Sekvencí | ~53,754 (ISTO kao RNN2) |
| **LSTM Lagen** | **4** (1 više nego RNN2!) |
| Learning Rate (Start) | 0.001 (ISTO kao RNN2) |
| LR Scheduler | ✅ DA (ReduceLROnPlateau) |
| Scheduler Patience | 10 epoha |
| Early Stopping Patience | 20 epoha |
| **Očekivana RMSE ex_22** | **TBD nakon treniranja** |

---

## 🎯 Eksperiment

**Fragestellung:** Može li dublja mreža (više LSTM slojeva) učiti kompleksnije obrasce?

---

## ⚗️ Hipoteze

✅ **Optimistička:** Više kapaciteta = bolja generalizacija
❌ **Pesimistička:** Overfitting na training data = loši rezultati na test-u

---

## 🏗️ Arhitektura

```
RNN2 (3 sloja):
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → Dense(1) → Output

RNN4 (4 sloja):
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → LSTM4(32) → Dense(1) → Output
                                                   ↑↑↑↑ NOVI SLOJ!
```

---

## 📊 Kompleksnost

- RNN2: ~3,105 parametara
- RNN4: ~4,145 parametara (+33% više!)

---

## 🔧 Datoteke

- **einfacheRNN4_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file

---

## 🚀 Kako Trenirati

1. Otvori Google Colab
2. Kopira `einfacheRNN4_Colab.py`
3. Shift+Enter
4. Učitaj 6 CSV-a
5. Čekaj ~50 min (malo duže nego RNN2 jer je dublji)
6. Download `einfachesRNN4.pth`
7. Poredi s RNN2 - da li je globlja mreža bolji?

---

**Kreirano:** 2026-07-01  
**Eksperiment:** Network Depth Deeper  
**Status:** Spreman za treniranje

