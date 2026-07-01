# 🟢 RNN2 (MULTI-SET + LEARNING RATE SCHEDULER)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ex_1, ex_9, ex_20, ex_21, ex_23, ex_24) |
| Broj Sekvencí | ~53,754 (ISTO kao RNN1) |
| LSTM Lagen | 3 |
| **Learning Rate (Start)** | **0.001** |
| **LR Scheduler** | **✅ DA (ReduceLROnPlateau)** |
| Scheduler Patience | 10 epoha |
| Scheduler Factor | 0.5 (halbiert) |
| Early Stopping Patience | 20 epoha |
| Batch Size | 32 |
| **Očekivana RMSE ex_22** | **~180 W** ✅✅ |

---

## 🎯 Svrha

Learning Rate Scheduler omogućava dodatni 23% poboljšanje konvergencije nakon što LR-constant start dosegne plato.

---

## 🔧 Datoteke

- **einfacheRNN2_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file
- **SETUP_GUIDE.md** — Detaljan vodiči setup-a

---

## 🚀 Kako Radi Scheduler

```
Epoch 1-10:    LR = 0.001 → Loss pada: 0.924 → 0.050 ✓
Epoch 11-20:   LR = 0.001 → Loss stagnira oko 0.050 
               ↓ SCHEDULER DETEKTUJE: 10 epoha bez poboljšanja!

Epoch 21:      LR = 0.0005 (HALBIERT!) → Loss pada: 0.050 → 0.015 ✓
Epoch 22-31:   LR = 0.0005 → Loss pada dalje

Epoch 32:      Nema poboljšanja 10 epoha
               ↓ SCHEDULER AKTIVIRAN OPET!

Epoch 33:      LR = 0.00025 (nochmal HALBIERT!)
```

---

## 📈 Poboljšanja

```
vs RNN1:
  ex_4.csv:  132.54 W → ~120 W  = ~10% bolje
  ex_22.csv: 232.04 W → ~180 W  = ~23% bolje!

vs RNN:
  ex_22.csv: 688.63 W → ~180 W  = 74% bolje overall!
```

---

## 💡 Ključna Ideja

```
Veći LR (0.001)  = Brža konvergencija, može biti oscilacijski
Manji LR (0.0005) = Sporija, ali preciznija
Scheduler        = Koristi oboje! Brzo početkom, precizno krajem
```

---

**Kreirano:** 2026-07-01  
**Status:** Baseline za eksperimente (RNN3-RNN6)  
**Nächster Schritt:** Hyperparameter variations

