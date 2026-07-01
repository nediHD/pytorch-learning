# 🟢 RNN5 (MULTI-SET + SCHEDULER + 2 LSTM SLOJA)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ISTO kao RNN2) |
| Broj Sekvencí | ~53,754 (ISTO kao RNN2) |
| **LSTM Lagen** | **2** (1 manje nego RNN2!) |
| Learning Rate (Start) | 0.001 (ISTO kao RNN2) |
| LR Scheduler | ✅ DA (ReduceLROnPlateau) |
| Scheduler Patience | 10 epoha |
| Early Stopping Patience | 20 epoha |
| **Očekivana RMSE ex_22** | **TBD nakon treniranja** |

---

## 🎯 Eksperiment

**Fragestellung:** Jesu li 3 LSTM sloja zaista potrebna, ili je 2 dovoljna?

---

## ⚗️ Hipoteze

✅ **Optimistička:** Jednostavniji model = dovoljno kapaciteta, brže treniranje, nema overfitting-a
❌ **Pesimistička:** Premalo kapaciteta = loši RMSE, underfitting

---

## 🏗️ Arhitektura

```
RNN2 (3 sloja):
  Input(1) → LSTM1(32) → LSTM2(32) → LSTM3(32) → Dense(1) → Output

RNN5 (2 sloja):
  Input(1) → LSTM1(32) → LSTM2(32) → Dense(1) → Output
                                      ↑ DIREKTNO NA OUTPUT (nema LSTM3)
```

---

## 📊 Kompleksnost

- RNN2: ~3,105 parametara (sporije)
- RNN5: ~2,065 parametara (-33%, brže!)

**Prednost:** Ako RNN5 daje iste rezultate kao RNN2, može se koristiti za brže treniranje!

---

## 🔧 Datoteke

- **einfacheRNN5_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file

---

## 🚀 Kako Trenirati

1. Otvori Google Colab
2. Kopira `einfacheRNN5_Colab.py`
3. Shift+Enter
4. Učitaj 6 CSV-a
5. Čekaj ~30 min (brže nego RNN2!)
6. Download `einfachesRNN5.pth`
7. Poredi s RNN2 - jednaki rezultati ali brže?

---

**Kreirano:** 2026-07-01  
**Eksperiment:** Network Depth Shallower  
**Status:** Spreman za treniranje

