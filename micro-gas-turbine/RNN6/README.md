# 🟠 RNN6 (MULTI-SET + SCHEDULER + NIŽI LR)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ISTO kao RNN2) |
| Broj Sekvencí | ~53,754 (ISTO kao RNN2) |
| LSTM Lagen | 3 (ISTO kao RNN2) |
| **Learning Rate (Start)** | **0.0005** (2x NIŽI od RNN2!) |
| LR Scheduler | ✅ DA (ReduceLROnPlateau) |
| Scheduler Patience | 10 epoha |
| Early Stopping Patience | 20 epoha |
| **Očekivana RMSE ex_22** | **TBD nakon treniranja** |

---

## 🎯 Eksperiment

**Fragestellung:** Bolji li konzervativan početak (0.0005) od balansiranog (0.001)?

---

## ⚗️ Hipoteze

✅ **Optimistička:** Preciznije početno učenje = bolja fine-tuning faza = bolji RMSE
❌ **Pesimistička:** Previše sporo na početku = verspätete Konvergenz

---

## 🔧 Datoteke

- **einfacheRNN6_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file

---

## 📊 LR Spektrum

```
RNN3:  0.01   ← AGRESIVNO (10x veće)
RNN2:  0.001  ← BALANCED (baseline)
RNN6:  0.0005 ← KONZERVATIVNO (2x manje)
```

**Ovo je suprotni kraj od RNN3 — testiram cijeli spektar LR vrijednosti!**

---

## 🚀 Kako Trenirati

1. Otvori Google Colab
2. Kopira `einfacheRNN6_Colab.py`
3. Shift+Enter
4. Učitaj 6 CSV-a
5. Čekaj ~40 min
6. Download `einfachesRNN6.pth`
7. Poredi s RNN2 i RNN3 - koji LR je najbolji?

---

**Kreirano:** 2026-07-01  
**Eksperiment:** Learning Rate Lower  
**Status:** Spreman za treniranje

