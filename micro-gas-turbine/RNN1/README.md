# 🟡 RNN1 (MULTI-SET + EARLY STOPPING)

---

## 📊 Model Specification

| Parametar | Vrijednost |
|-----------|-----------|
| Training Datasets | 6 (ex_1, ex_9, ex_20, ex_21, ex_23, ex_24) |
| Broj Sekvencí | ~53,754 (5.7x više nego RNN!) |
| LSTM Lagen | 3 |
| Learning Rate | 0.001 (konstant) |
| LR Scheduler | ❌ Nein |
| Early Stopping Patience | 20 epoha |
| Batch Size | 32 |
| **Očekivana RMSE ex_22** | **232.04 W** ✅ |

---

## 🎯 Svrha

Multi-dataset training pokazuje moć datavarijabitilnosti. RNN1 je **66% bolji** od RNN-a!

---

## 🔧 Datoteke

- **einfacheRNN_Colab.py** (u parent direktoriju `../`)
- **README.md** — Ovaj file

---

## 📝 Kako Koristiti

1. Otvori Google Colab
2. Kopira kod iz `../einfacheRNN_Colab.py`
3. Zalijepi u Colab cell
4. Udi u Shift+Enter
5. Učitaj 6 CSV datoteke
6. Čekaj ~50 minuta za GPU trening
7. Download `einfachesRNN1.pth`

---

## 📊 Poboljšanja vs RNN

```
ex_4.csv:  384.19 W → 132.54 W  = 65% BOLJE! ✅✅✅
ex_22.csv: 688.63 W → 232.04 W  = 66% BOLJE! ✅✅✅
```

---

## 💡 Lekcija

**Datavariabilnost je kritična za generalizaciju!**

Umjesto treniranja samo na ex_1.csv, korištenje 6 raznih datoteka omogućava modelu da nauči generalne obrasce koji se primjenjuju na sve dataset-e.

---

**Kreirano:** 2026-07-01  
**Status:** Multi-set baseline - prije LR Scheduler eksperimenta

