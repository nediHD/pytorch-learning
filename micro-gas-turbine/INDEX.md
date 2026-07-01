# 📚 MASTER INDEX - Gas Turbine RNN Projekt

**Centralna dokumentacija za sve RNN modele i eksperimente**

---

## 🗂️ STRUKTURA FOLDERA

```
micro-gas-turbine/
├── INDEX.md                          ← TI SI OVDJE
│
├── RNN/                              ← 🟠 BASELINE (1 dataset)
│   ├── README.md
│   └── einfacheRNN_Colab.py (u parent)
│
├── RNN1/                             ← 🟡 MULTI-SET (6 datasets)
│   ├── README.md
│   └── einfacheRNN_Colab.py (u parent)
│
├── RNN2/                             ← 🟢 SCHEDULER (baseline za eksperimente)
│   ├── README.md
│   └── einfacheRNN2_Colab.py (u parent)
│
├── RNN3/                             ← 🔵 EKSPERIMENT: Viši LR (0.01)
│   ├── README.md
│   └── einfacheRNN3_Colab.py (u parent)
│
├── RNN4/                             ← 🟣 EKSPERIMENT: 4 LSTM sloja
│   ├── README.md
│   └── einfacheRNN4_Colab.py (u parent)
│
├── RNN5/                             ← 🟢 EKSPERIMENT: 2 LSTM sloja
│   ├── README.md
│   └── einfacheRNN5_Colab.py (u parent)
│
├── RNN6/                             ← 🟠 EKSPERIMENT: Niži LR (0.0005)
│   ├── README.md
│   └── einfacheRNN6_Colab.py (u parent)
│
├── shared/                           ← Zajedničke datoteke
│   ├── TRAINING_LOG.md
│   ├── MODEL_GUIDE_DE.md
│   ├── MODELS_COMPARISON_DE.md
│   └── RNN_EXPERIMENTS_SUMMARY_DE.md
│
├── data/                             ← Training/Test podatci
│   ├── train/
│   │   ├── ex_1.csv
│   │   ├── ex_9.csv
│   │   ├── ex_20.csv
│   │   ├── ex_21.csv
│   │   ├── ex_23.csv
│   │   └── ex_24.csv
│   └── test/
│       ├── ex_4.csv
│       └── ex_22.csv
│
├── test_model.py                     ← Testiraj sve modele
└── plot_modeli.py                    ← Vizualizacija svih modela
```

---

## 🎯 BRZI NAVIGACIJSKI VODIČI

### Za Brzo Razumijevanje Svega

**Htješ li znati:**

| Pitanje | Gdje Ići | Datoteka |
|---------|---------|---------|
| Što je razlika između RNN i RNN1? | RNN/ ili RNN1/ | README.md |
| Kako radi Learning Rate Scheduler? | RNN2/ ili shared/ | README.md ili MODEL_GUIDE_DE.md |
| Što je RNN3, RNN4, RNN5, RNN6? | RNN3-6/ ili shared/ | README.md ili MODELS_COMPARISON_DE.md |
| Kako trenirati koji model? | RNN*/RNN* | README.md |
| Svi hiperparametri (detaljno) | shared/ | TRAINING_LOG.md |
| Kako testirati sve modele? | Parent direktorij | test_model.py |
| Kako vizualizirati rezultate? | Parent direktorij | plot_modeli.py |

---

## 📊 PREGLED SVIH MODELA

### Progresija Poboljšanja

```
RNN (Baseline - 1 dataset)
  RMSE ex_22: 688.63 W ❌
  
RNN1 (Multi-set - 6 datasets)
  RMSE ex_22: 232.04 W ✅ (66% bolje!)
  
RNN2 (Multi-set + Scheduler)
  RMSE ex_22: ~180 W ✅✅ (23% dodatno bolje!)
  
RNN3 (Viši LR: 0.01)
  RMSE ex_22: TBD → ? (agresivnije učenje?)
  
RNN4 (4 LSTM sloja)
  RMSE ex_22: TBD → ? (dublja mreža?)
  
RNN5 (2 LSTM sloja)
  RMSE ex_22: TBD → ? (plića mreža?)
  
RNN6 (Niži LR: 0.0005)
  RMSE ex_22: TBD → ? (konzervativan početak?)
```

---

## 🚀 KAKO POČETI

### Korak 1: Razumijevanje
```
1. Čitaj INDEX.md (ovaj file)
2. Čitaj shared/MODELS_COMPARISON_DE.md (detaljan opis)
3. Čitaj RNN2/README.md (baseline za eksperimente)
```

### Korak 2: Treniranje U Google Colab
```
Za svaki model (RNN3-RNN6):
  1. Otvori Google Colab: colab.research.google.com
  2. Kreiraj novu Notebook
  3. Aktiviraj GPU: Runtime → Change runtime → GPU
  4. Kopira kod iz RNN*/einfacheRNN*_Colab.py
  5. Zalijepi u Colab cell
  6. Shift+Enter (training počinje)
  7. Učitaj CSV datoteke
  8. Čekaj 30-50 minuta
  9. Download .pth datoteke
```

### Korak 3: Testiranje Rezultata
```bash
# Nakon što imaš sve .pth datoteke:
python test_model.py        # Vidi RMSE za sve modele
python plot_modeli.py       # Generiraj prediction plots
```

---

## 📁 KOLIKO DATOTEKA JE GDJE?

| Lokacija | Sadržaj |
|----------|---------|
| RNN/ | 1 README |
| RNN1/ | 1 README |
| RNN2/ | 1 README |
| RNN3/ | 1 README |
| RNN4/ | 1 README |
| RNN5/ | 1 README |
| RNN6/ | 1 README |
| shared/ | 4 dokumentacijske datoteke |
| parent dir | 2 test skripte + 1 INDEX |

**Total: 7 README + 4 shared docs + 2 test scripts + 1 INDEX = 14 dokumentacijskih datoteka**

---

## 🔧 TEHNIČKI DETALJI

### Svi RNN Modeli Koriste

- ✅ Multi-set Training (6 datoteka)
- ✅ Learning Rate Scheduler (RNN2+)
- ✅ Early Stopping (sve)
- ✅ 100% Nemačka Dokumentacija
- ✅ Google Colab Optimiziran Kod
- ✅ GPU Support (CUDA/CPU fallback)

### Datoteke Po Tipu

```
Colab Python Skripte:
  - einfacheRNN_Colab.py (RNN + RNN1)
  - einfacheRNN2_Colab.py (RNN2)
  - einfacheRNN3_Colab.py (RNN3)
  - einfacheRNN4_Colab.py (RNN4)
  - einfacheRNN5_Colab.py (RNN5)
  - einfacheRNN6_Colab.py (RNN6)

Model README-ovi:
  - RNN/README.md
  - RNN1/README.md
  - RNN2/README.md
  - RNN3/README.md
  - RNN4/README.md
  - RNN5/README.md
  - RNN6/README.md

Zajedničke Dokumemenacije:
  - shared/TRAINING_LOG.md (detalje sve hiperparametre)
  - shared/MODEL_GUIDE_DE.md (usporedbe)
  - shared/MODELS_COMPARISON_DE.md (eksperimentalni design)
  - shared/RNN_EXPERIMENTS_SUMMARY_DE.md (brzi vodiči)

Test/Viz Skripte:
  - test_model.py (test sve modele)
  - plot_modeli.py (vizualizacija)

Podaci:
  - data/train/ (6 CSV datoteka)
  - data/test/ (2 CSV datoteka)
```

---

## ❓ ČESTE PITANJA

**P: Gdje trebam početi?**
O: Čitaj RNN2/README.md → shared/MODELS_COMPARISON_DE.md → kreni s trenianjem RNN3-6 u Colabu

**P: Gdje je kod za RNN i RNN1?**
O: Oba su u `einfacheRNN_Colab.py` (parent direktorij). Pogledaj RNN/README.md i RNN1/README.md

**P: Gdje su svi Colab skripti?**
O: Svi su u parent direktoriju (`einfacheRNN*_Colab.py`)

**P: Trebam li čitati sve dokumentacije?**
O: Preporučeno:
  - Brz pregled: INDEX.md → RNN*/README.md
  - Detaljno: shared/MODELS_COMPARISON_DE.md

**P: Kako znati koji model je najbolji?**
O: Pokrenite `python test_model.py` nakon treniranja - dat će vam ranking po RMSE

**P: Što trebam podaktteti za treniranje?**
O: Samo 6 CSV datoteka iz `data/train/` - kod će ih automatski učitati u Colabu

---

## 🎓 EDUKACIJSKE LEKCIJE PO MODELU

| Model | Učiš | Zaključak |
|-------|-----|----------|
| RNN | Baseline + Overfitting | 1 dataset nije dovoljno |
| RNN1 | Multi-dataset snaga | Datavariabilnost kritična |
| RNN2 | Adaptive LR | Scheduler poboljšava finu tuning |
| RNN3 | LR spektrum (više) | Je li agresivno bolje? |
| RNN4 | Depth experiment (više) | Trebam li 4 sloja? |
| RNN5 | Depth experiment (manje) | Trebam li samo 2 sloja? |
| RNN6 | LR spektrum (manje) | Je li konzervativan bolje? |

---

## 📈 SLJEDEĆI KORACI NAKON EKSPERIMENATA

1. **Analiza:** Viditi koja konfiguracija daje najbolje RMSE
2. **Optimizacija:** Fine-tune najbolji model
3. **Knowledge-Guided Learning:** Dodaj Permissible States ograničenja
4. **Produkcija:** Koristi najbolji model za predikcije

---

## 📅 VREMENSKE CRTE

| Što | Gdje | Vrijeme |
|-----|------|---------|
| Setup (čitanje) | INDEX + README | 10-15 min |
| Treniranje RNN3 | Google Colab | 40-50 min |
| Treniranje RNN4 | Google Colab | 50 min (duže) |
| Treniranje RNN5 | Google Colab | 30-40 min (brže) |
| Treniranje RNN6 | Google Colab | 40-50 min |
| Testing (svi modeli) | Lokalno | 5-10 min |
| **TOTAL** | | **3-4 sata** |

---

## 🎯 KONAČNI CILJ

Kroz ove 4 eksperimenta (RNN3, RNN4, RNN5, RNN6) razumiješ:
- 💡 Impact của Learning Rate
- 💡 Impact của Network Depth
- 💡 Kako odabrati najbolje hiperparametre
- 💡 Kako funkcionira Learning Rate Scheduler
- 💡 Kako mjeriti model performance (RMSE, MAE)

Tada možeš sigurno prelaziti na **Knowledge-Guided Learning** fazu s Permissible States.

---

**INDEX Kreirano:** 2026-07-01  
**Verzija:** 1.0  
**Zadnja ažuriranja:** Sve folderе i README-ovi dostupni

---

## 🚀 READY TO BEGIN?

Kreni s **RNN2/README.md** → **shared/MODELS_COMPARISON_DE.md** → Treniranje u Google Colabu! 🎉

