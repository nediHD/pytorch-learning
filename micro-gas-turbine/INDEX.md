# 📚 MASTER INDEX - Gas Turbine RNN Projekt

**Centralna dokumentacija i navigacija za sve RNN modele i eksperimente**

---

## 🗂️ KOMPLETAN DIREKTORIJ STRUKTURE

```
micro-gas-turbine/
│
├── 📄 INDEX.md                          ← 🎯 TI SI OVDJE - MASTER VODIČI
├── 📄 README.md                         ← Main projekt README
│
├── 📁 RNN/                              ← 🟠 BASELINE (1 dataset, RMSE 688W)
│   ├── 📄 README.md                    ← RNN specifikacije
│   ├── 📄 einfacheRNN_Colab.py         ← Training skript
│   ├── 📁 models/                      ← Model weights
│   │   └── 📄 einfachesRNN.pth        ← Trenirani model
│   └── 📁 results/                     ← Predviđanja i plotovi
│       ├── 📄 plot_einfachesRNN_ex_22.png
│       └── 📄 plot_einfachesRNN_ex_4.png
│
├── 📁 RNN1/                             ← 🟡 MULTI-SET (6 datasets, RMSE 232W)
│   ├── 📄 README.md                    ← RNN1 specifikacije
│   ├── 📄 einfacheRNN_Colab.py         ← Training skript (ISTO kao RNN)
│   ├── 📁 models/                      ← Model weights
│   │   └── 📄 einfachesRNN1.pth       ← Trenirani model
│   └── 📁 results/                     ← Predviđanja i plotovi
│       ├── 📄 plot_einfachesRNN1_ex_22.png
│       └── 📄 plot_einfachesRNN1_ex_4.png
│
├── 📁 RNN2/                             ← 🟢 SCHEDULER (BASELINE, ~180W RMSE)
│   ├── 📄 README.md                    ← RNN2 specifikacije
│   ├── 📄 einfacheRNN2_Colab.py        ← Training skript (Learning Rate Scheduler)
│   ├── 📁 models/                      ← Model weights (čeka treniranje)
│   │   └── .gitkeep
│   └── 📁 results/                     ← Rezultati (čeka treniranje)
│       └── .gitkeep
│
├── 📁 RNN3/                             ← 🔵 EKSPERIMENT: Viši LR (0.01)
│   ├── 📄 README.md                    ← RNN3 specifikacije
│   ├── 📄 einfacheRNN3_Colab.py        ← Training skript (LR=0.01)
│   ├── 📁 models/                      ← Model weights (čeka treniranje)
│   │   └── .gitkeep
│   └── 📁 results/                     ← Rezultati (čeka treniranje)
│       └── .gitkeep
│
├── 📁 RNN4/                             ← 🟣 EKSPERIMENT: 4 LSTM sloja
│   ├── 📄 README.md                    ← RNN4 specifikacije
│   ├── 📄 einfacheRNN4_Colab.py        ← Training skript (Dublja arhitektura)
│   ├── 📁 models/                      ← Model weights (čeka treniranje)
│   │   └── .gitkeep
│   └── 📁 results/                     ← Rezultati (čeka treniranje)
│       └── .gitkeep
│
├── 📁 RNN5/                             ← 🟢 EKSPERIMENT: 2 LSTM sloja
│   ├── 📄 README.md                    ← RNN5 specifikacije
│   ├── 📄 einfacheRNN5_Colab.py        ← Training skript (Plića arhitektura)
│   ├── 📁 models/                      ← Model weights (čeka treniranje)
│   │   └── .gitkeep
│   └── 📁 results/                     ← Rezultati (čeka treniranje)
│       └── .gitkeep
│
├── 📁 RNN6/                             ← 🟠 EKSPERIMENT: Niži LR (0.0005)
│   ├── 📄 README.md                    ← RNN6 specifikacije
│   ├── 📄 einfacheRNN6_Colab.py        ← Training skript (LR=0.0005)
│   ├── 📁 models/                      ← Model weights (čeka treniranje)
│   │   └── .gitkeep
│   └── 📁 results/                     ← Rezultati (čeka treniranje)
│       └── .gitkeep
│
├── 📁 shared/                           ← 📚 ZAJEDNIČKE DATOTEKE
│   ├── 📁 docs/                        ← Dokumentacija
│   │   ├── 📄 TRAINING_LOG.md         ← Svi hiperparametri detalje
│   │   ├── 📄 MODEL_GUIDE_DE.md       ← Usporedne tablice i vodiči
│   │   ├── 📄 MODELS_COMPARISON_DE.md ← Eksperimentalni dizajn
│   │   ├── 📄 RNN_EXPERIMENTS_SUMMARY_DE.md ← Brzi scenariji
│   │   └── 📄 RNN2_SETUP.md           ← Scheduler vodiči
│   ├── 📁 references/                  ← Strane reference
│   │   └── 📄 Knowledge-Guided Learning of Temporal Dynamics...pdf
│   └── 📁 tools/                       ← Test i evaluacijski alati
│       ├── 📄 test_model.py            ← Testiraj sve modele na RMSE
│       └── 📄 plot_modeli.py           ← Generiraj prediction plots
│
├── 📁 data/                             ← TRAINI/TEST PODATCI
│   ├── 📁 train/                       ← 6 datoteka za treniranje
│   │   ├── 📄 ex_1.csv
│   │   ├── 📄 ex_9.csv
│   │   ├── 📄 ex_20.csv
│   │   ├── 📄 ex_21.csv
│   │   ├── 📄 ex_23.csv
│   │   └── 📄 ex_24.csv
│   └── 📁 test/                        ← 2 datoteke za evaluaciju
│       ├── 📄 ex_4.csv
│       └── 📄 ex_22.csv
│
├── 📄 einfacheRNN_Colab.py              ← Python: RNN + RNN1 (parent za kopiranje)
├── 📄 einfacheRNN2_Colab.py             ← Python: RNN2 (parent za kopiranje)
├── 📄 einfacheRNN3_Colab.py             ← Python: RNN3 (parent za kopiranje)
├── 📄 einfacheRNN4_Colab.py             ← Python: RNN4 (parent za kopiranje)
├── 📄 einfacheRNN5_Colab.py             ← Python: RNN5 (parent za kopiranje)
└── 📄 einfacheRNN6_Colab.py             ← Python: RNN6 (parent za kopiranje)
```

---

## 🎯 GDJE JE ŠTA? (BRZI NAVIGACIJSKI VODIČI)

| Trebam | Gdje Ići | Datoteka |
|--------|---------|----------|
| **Razumijevanje jednog modela** | RNN*/README.md | Specifikacije, hiperparametri, brzi vodiči |
| **Svi hiperparametri detaljno** | shared/docs/ | TRAINING_LOG.md |
| **Usporedbe između modela** | shared/docs/ | MODEL_GUIDE_DE.md ili MODELS_COMPARISON_DE.md |
| **Kako trenirati** | RNN*/README.md → einfacheRNN*_Colab.py | Python skript za treniranje |
| **Gdje je trenirani model** | RNN*/models/ | einfachesRNN*.pth (nakon treniranja) |
| **Gdje su plotovi rezultata** | RNN*/results/ | plot_einfachesRNN*_*.png |
| **Kako testirati sve modele** | shared/tools/ | test_model.py |
| **Kako vizualizirati** | shared/tools/ | plot_modeli.py |
| **Referentni paper** | shared/references/ | Knowledge-Guided Learning PDF |
| **Brzi scenariji i FAQ** | shared/docs/ | RNN_EXPERIMENTS_SUMMARY_DE.md |

---

## 📂 DATOTEKE PO TIPU

### Python Training Skripti (Colab)
```
einfacheRNN_Colab.py      → RNN + RNN1 (ista skripta)
einfacheRNN2_Colab.py     → RNN2 (Scheduler)
einfacheRNN3_Colab.py     → RNN3 (Viši LR)
einfacheRNN4_Colab.py     → RNN4 (4 LSTM)
einfacheRNN5_Colab.py     → RNN5 (2 LSTM)
einfacheRNN6_Colab.py     → RNN6 (Niži LR)
```

### Model Weights (.pth)
```
RNN/models/einfachesRNN.pth       → Trenirani RNN model
RNN1/models/einfachesRNN1.pth     → Trenirani RNN1 model
RNN2/models/einfachesRNN2.pth     → (čeka treniranje)
RNN3/models/einfachesRNN3.pth     → (čeka treniranje)
RNN4/models/einfachesRNN4.pth     → (čeka treniranje)
RNN5/models/einfachesRNN5.pth     → (čeka treniranje)
RNN6/models/einfachesRNN6.pth     → (čeka treniranje)
```

### Plot Rezultati (.png)
```
RNN/results/plot_einfachesRNN_ex_*.png      → RNN vizualizacija
RNN1/results/plot_einfachesRNN1_ex_*.png    → RNN1 vizualizacija
RNN2-6/results/                             → (čeka treniranje)
```

### Dokumentacijske Datoteke (.md)
```
RNN*/README.md                          → Specifikacije svakog modela
shared/docs/TRAINING_LOG.md             → Svi hiperparametri detalje
shared/docs/MODEL_GUIDE_DE.md           → Usporedne tablice
shared/docs/MODELS_COMPARISON_DE.md     → Eksperimentalni dizajn
shared/docs/RNN_EXPERIMENTS_SUMMARY_DE.md → Brzi scenariji
shared/docs/RNN2_SETUP.md               → Scheduler vodiči
```

### Testiranje i Evaluacija
```
shared/tools/test_model.py              → Testiraj sve 7 modela
shared/tools/plot_modeli.py             → Generiraj sve plotove
```

### Reference
```
shared/references/Knowledge-Guided...pdf → Originalni paper
```

---

## 🚀 KAKO POČETI

### KORAK 1: Razumijevanje (15 minuta)
```
1. Čitaj INDEX.md (ovaj file) - pregled strukture
2. Čitaj RNN/README.md - razumijevanje baseline
3. Čitaj RNN2/README.md - razumijevanje Scheduler-a
4. Čitaj shared/docs/MODELS_COMPARISON_DE.md - svi eksperimenti obješnjeni
```

### KORAK 2: Treniranje Modela (3-4 sata GPU)
```
Za svaki model (RNN3, RNN4, RNN5, RNN6):
  1. Otvori RNN*/README.md
  2. Kopira kod iz einfacheRNN*_Colab.py
  3. Otvori Google Colab: colab.research.google.com
  4. Runtime → Change runtime → GPU
  5. Zalijepi kod, Shift+Enter
  6. Učitaj 6 CSV datoteke iz data/train/
  7. Čekaj 30-50 minuta
  8. Preuzmi einfachesRNN*.pth
  9. Postavi u RNN*/models/
```

### KORAK 3: Testiranje (10 minuta)
```
1. cd shared/tools/
2. python test_model.py      # Vidi RMSE sve 7 modela
3. python plot_modeli.py     # Generiraj plotove
4. Pogledaj rezultate u RNN*/results/
```

### KORAK 4: Analiza
```
1. Otvorite shared/docs/TRAINING_LOG.md
2. Dopuní RMSE vrijednosti koje ste izmjerili
3. Usporedi s očekivanima
4. Kreiraj zaključke za sljedeću fazu
```

---

## 📊 PROGRESIJA POBOLJŠANJA

```
RNN          (1 dataset)  → RMSE: 688.63 W ❌
    ↓ (4.3x vise podatka)
RNN1         (6 datasets) → RMSE: 232.04 W ✅ (66% bolje!)
    ↓ (Scheduler)
RNN2         (scheduler)  → RMSE: ~180 W ✅✅ (23% dalje)
    ↙        ↙       ↘        ↘
RNN3      RNN4        RNN5     RNN6
(0.01LR)  (4 sloja)  (2 sloja) (0.0005LR)
TBD?      TBD?        TBD?     TBD?
```

---

## ⚙️ TEHNIČKI DETALJI

### Sve Modele Koriste
- ✅ Multi-set Training (6 CSV datoteke)
- ✅ Learning Rate Scheduler (RNN2+)
- ✅ Early Stopping (patience=20)
- ✅ 100% Nemačka Dokumentacija
- ✅ Google Colab Optimiziran Kod
- ✅ GPU Support (CUDA/CPU fallback)

### Razlike Između Modela
```
RNN2:  LR=0.001, 3 LSTM, Baseline
RNN3:  LR=0.01,  3 LSTM, Eksperiment: viši LR
RNN4:  LR=0.001, 4 LSTM, Eksperiment: dublja
RNN5:  LR=0.001, 2 LSTM, Eksperiment: plića
RNN6:  LR=0.0005, 3 LSTM, Eksperiment: niži LR
```

---

## ❓ ČESTE PITANJA

**P: Gdje trebam početi?**  
O: INDEX.md (ovdje) → RNN2/README.md → shared/docs/MODELS_COMPARISON_DE.md

**P: Gdje su Colab skripti?**  
O: einfacheRNN*_Colab.py (u parent direktoriju i u RNN*/lokalno)

**P: Gdje je kod za RNN i RNN1?**  
O: Oboje su u einfacheRNN_Colab.py (ista skripta, drugačitji dataset loading)

**P: Trebam li sve čitati?**  
O: Ne - RNN*/README.md je dovoljno za brz početak

**P: Koji model trebam trenirati?**  
O: Trebaj RNN2-6 u Colab-u (RNN i RNN1 su već trenirani)

**P: Gdje će biti .pth datoteke?**  
O: U RNN*/models/ nakon što preuzmete s Colab-a

**P: Gdje će biti plotovi?**  
O: U RNN*/results/ nakon što pokrenete plot_modeli.py

**P: Trebam li data/ folder?**  
O: DA - trebaju data/train/ i data/test/ CSV datoteke

---

## 🎓 EDUKACIJSKE LEKCIJE

| Model | Učiš | Zaključak |
|-------|------|----------|
| RNN | Baseline + Overfitting | 1 dataset nije dovoljno |
| RNN1 | Multi-dataset Snaga | Datavariabilnost kritična |
| RNN2 | Adaptive LR | Scheduler dramatično pomaže |
| RNN3 | Viši LR | Agresivniji početak? |
| RNN4 | Dublja Arhitektura | Trebam li više slojeva? |
| RNN5 | Plića Arhitektura | Trebam li samo 2 sloja? |
| RNN6 | Niži LR | Konzervativan početak? |

---

## 📈 VREMENSKI OKVIR

| Aktivnost | Gdje | Vrijeme |
|----------|------|---------|
| Setup (čitanje) | Index + README | 10-15 min |
| Treniranje RNN2 | Colab GPU | 40-50 min |
| Treniranje RNN3 | Colab GPU | 40-50 min |
| Treniranje RNN4 | Colab GPU | 50 min (duže) |
| Treniranje RNN5 | Colab GPU | 30-40 min (brže) |
| Treniranje RNN6 | Colab GPU | 40-50 min |
| Testing (svi) | Lokalno | 5-10 min |
| **TOTAL** | - | **4-5 sati** |

---

## 🔄 WORKFLOW

```
1. Čitaj dokumentaciju
   └─ INDEX.md → RNN*/README.md → shared/docs/

2. Treniraj modele
   └─ Google Colab: Copy einfacheRNN*_Colab.py → Train → Download .pth

3. Testiraj
   └─ shared/tools/test_model.py → shared/tools/plot_modeli.py

4. Analiziraj
   └─ Usporedi RMSE → Update TRAINING_LOG.md → Zaključi

5. Naprijed
   └─ Knowledge-Guided Learning s Permissible States
```

---

## 🎯 KONAČNI CILJ

Kroz ove eksperimente razumiješ:
- 💡 Learning Rate Impact (LR spektrum: RNN3 vs RNN2 vs RNN6)
- 💡 Network Depth Impact (arhitektura: RNN4 vs RNN2 vs RNN5)
- 💡 Hyperparameter Tuning (kako odabrati najbolje)
- 💡 Model Evaluation (RMSE, MAE, MSE metrike)
- 💡 Data Importance (multi-dataset vs single)

Tada možeš sigurno prelaziti na **Knowledge-Guided Learning** fazu!

---

**INDEX Kreirano:** 2026-07-01  
**Zadnja Ažuriranja:** Kompletna reorganizacija datoteka u folderе  
**Verzija:** 2.0 (Reorganized Structure)

---

## 🚀 READY TO BEGIN?

1. **Brzo:** RNN*/README.md → Treniranje
2. **Detaljno:** INDEX → shared/docs/ → Treniranje
3. **Kompleto:** Svi README → Sva dokumentacija → Treniranje

**Let's go! 🎉**

