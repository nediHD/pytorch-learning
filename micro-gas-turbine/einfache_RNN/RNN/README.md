# 🟠 RNN (BASELINE - ORIGINAL)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Dataset | 1 (ex_1.csv samo) |
| Broj Sekvencí | 9,469 |
| LSTM Lagen | 3 |
| Learning Rate | 0.001 (konstant) |
| LR Scheduler | ❌ Nein |
| Early Stopping Patience | 20 epoha |
| Batch Size | 32 |
| **Očekivana RMSE ex_22** | **688.63 W** ❌ |

---

## 🎯 Svrha

Baseline model koji pokazuje problem overfitting-a kada se trenira samo na jednom dataset-u.

---

## 🔧 Datoteke

- **einfacheRNN_Colab.py** — Google Colab training skript
- **README.md** — Ovaj file

---

## 📌 Ključne Razlike od RNN1

```
RNN:  1 dataset  → 9,469 sekvencí    → RMSE 688 W ❌
RNN1: 6 datasetá → 53,754 sekvencí   → RMSE 232 W ✅ (66% bolje!)
```

**Zaključak:** Multi-dataset training drastično poboljšava rezultate!

---

**Kreirano:** 2026-07-01  
**Status:** Baseline - za referensu

