# 🔵 RNN3 (MULTI-SET + SCHEDULER + VIŠI LR)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ISTO kao RNN2) |
| Broj Sekvencí | ~53,754 (ISTO kao RNN2) |
| LSTM Lagen | 3 (ISTO kao RNN2) |
| **Learning Rate (Start)** | **0.01** (10x VIŠI nego RNN2!) |
| LR Scheduler | ✅ DA (ReduceLROnPlateau) |
| Scheduler Patience | 10 epoha |
| Early Stopping Patience | 20 epoha |
| Batch Size | 32 |
| **Očekivana RMSE ex_22** | **TBD nakon treniranja** |

---

## 🎯 Eksperiment

**Fragestellung:** Omogućava li agresivniji početak (0.01) bržu konvergenciju ili šteti stabilnosti?

---

## ⚗️ Hipoteze

✅ **Optimistička:** Brži početak + scheduler fine-tuning = RMSE može biti još bolji nego RNN2
❌ **Pesimistička:** Viši LR uzrokuje oscilacije, konačni RMSE gori od RNN2

---

## 🔧 Datoteke

- **einfacheRNN3_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file

---

## 📊 LR Spektrum

```
RNN3:  0.01   ← AGRESIVNO (10x veće)
RNN2:  0.001  ← BALANCED (baseline)
RNN6:  0.0005 ← KONZERVATIVNO (2x manje)
```

---

## 🚀 Kako Trenirati

1. Otvori Google Colab
2. Kopira `einfacheRNN3_Colab.py`
3. Shift+Enter
4. Učitaj 6 CSV-a
5. Čekaj ~40 min
6. Download `einfachesRNN3.pth`
7. Poredi s RNN2 rezultatima

---

**Kreirano:** 2026-07-01  
**Eksperiment:** Learning Rate Higher  
**Status:** Spreman za treniranje

