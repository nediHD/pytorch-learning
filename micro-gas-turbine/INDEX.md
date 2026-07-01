# 📚 MASTER INDEX - Gas Turbine RNN Projekt

**Centralna dokumentacija za sve RNN modele i Knowledge-Guided Learning**

---

## 🗂️ KOMPLETAN DIREKTORIJ STRUKTURA

```
micro-gas-turbine/
│
├── 📄 INDEX.md                          ← 🎯 TI SI OVDJE
├── 📄 README.md                         ← Project overview
│
├── 📁 einfache_RNN/                     ← 🤖 BASELINE MODELI (bez prior knowledge)
│   ├── 📄 README.md                    ← Vodiči za ovaj folder
│   ├── 📁 RNN/                         ← Baseline (1 dataset)
│   ├── 📁 RNN1/                        ← Multi-set (6 datasets)
│   ├── 📁 RNN2/                        ← + Scheduler
│   ├── 📁 RNN3/                        ← + Viši LR
│   ├── 📁 RNN4/                        ← + 4 LSTM sloja
│   ├── 📁 RNN5/                        ← + 2 LSTM sloja
│   └── 📁 RNN6/                        ← + Niži LR
│       (svaki sadrži: README.md, einfacheRNN*_Colab.py, models/, results/)
│
├── 📁 RNN_with_prior_knowledge/         ← 🧠 KNOWLEDGE-GUIDED LEARNING
│   ├── 📄 README.md                    ← Vodiči za Knowledge-Guided Learning
│   ├── 📁 RNN_KGL_v1/                  ← Osnovni Knowledge-Guided model
│   ├── 📁 RNN_KGL_v2/                  ← Optimizovano s Permissible States
│   └── 📁 RNN_KGL_v3/                  ← Finalna verzija
│       (svaki će sadržavati: README.md, kod, models/, results/)
│
├── 📁 shared/                           ← 📚 ZAJEDNIČKE DATOTEKE
│   ├── 📁 docs/                        ← Sve dokumentacije
│   │   ├── TRAINING_LOG.md
│   │   ├── MODEL_GUIDE_DE.md
│   │   ├── MODELS_COMPARISON_DE.md
│   │   ├── RNN_EXPERIMENTS_SUMMARY_DE.md
│   │   └── RNN2_SETUP.md
│   ├── 📁 tools/                       ← Test i evaluacijski alati
│   │   ├── test_model.py
│   │   └── plot_modeli.py
│   └── 📁 references/                  ← Research papers
│       └── Knowledge-Guided Learning PDF
│
├── 📁 data/                             ← PODATCI
│   ├── 📁 train/                       ← 6 CSV za treniranje
│   └── 📁 test/                        ← 2 CSV za evaluaciju
│
└── .gitignore
```

---

## 🎯 BRZA NAVIGACIJA

### **KORAK 1: Baseline Modeli (einfache_RNN)**

```
Trebam razumijevanje baseline RNN modela?
  → einfache_RNN/README.md

Trebam trenirati neki RNN model?
  → einfache_RNN/RNN*/README.md
  → einfache_RNN/RNN*/einfacheRNN*_Colab.py

Trebam sve hiperparametre?
  → shared/docs/TRAINING_LOG.md
```

### **KORAK 2: Knowledge-Guided Learning (RNN_with_prior_knowledge)**

```
Trebam razumijevanje Knowledge-Guided Learning?
  → RNN_with_prior_knowledge/README.md
  → shared/references/Knowledge-Guided Learning PDF

Trebam trenirati Knowledge-Guided model?
  → RNN_with_prior_knowledge/RNN_KGL_v*/README.md
  → RNN_with_prior_knowledge/RNN_KGL_v*/kod.py
```

### **KORAK 3: Testiranje**

```
Trebam testiraj sve modele?
  → shared/tools/test_model.py

Trebam visualizirati rezultate?
  → shared/tools/plot_modeli.py
```

---

## 📊 FAZE PROJEKTA

### **Faza 1️⃣: Baseline Modeli (SADAŠNJA FAZA)**

```
einfache_RNN/ sadrži 7 progresivnih modela:

RNN:     688 W ❌ (baseline greška)
    ↓
RNN1:    232 W ✅ (66% bolje - multi-set)
    ↓
RNN2:    ~180 W ✅✅ (23% dalje - scheduler)
    ↓
RNN3-6:  TBD 🔬 (hyperparameter eksperimenti)

Cilj: Razumijevanje što pomaže poboljšati RMSE
```

### **Faza 2️⃣: Knowledge-Guided Learning (SLJEDEĆA FAZA)**

```
RNN_with_prior_knowledge/ sadrži modele s prior znanjem:

RNN_KGL_v1:  Bazna verzija + Permissible States
RNN_KGL_v2:  Optimizovana s prior knowledge loss
RNN_KGL_v3:  Finalna verzija (~140-150 W očekivano?)

Cilj: Kombinirati data-driven + knowledge-driven learning
```

---

## 🚀 KAKO POČETI

### **SCENARIO A: Trebam trenirati baseline modele**

1. Čitaj `einfache_RNN/README.md` - razumijevanje strukture
2. Odaberi model: `einfache_RNN/RNN*/README.md`
3. Kopiraj kod: `einfache_RNN/RNN*/einfacheRNN*_Colab.py`
4. Treniraj u Google Colab (30-50 minuta)
5. Testiraj: `shared/tools/test_model.py`
6. Vizualizacija: `shared/tools/plot_modeli.py`

### **SCENARIO B: Trebam razumijevanje Knowledge-Guided Learning**

1. Čitaj `RNN_with_prior_knowledge/README.md`
2. Čitaj paper: `shared/references/Knowledge-Guided Learning PDF`
3. Čekaj na RNN_KGL_v* skripte koje će biti kreirane

### **SCENARIO C: Trebam sve informacije**

1. Čitaj: `einfache_RNN/README.md` (baseline)
2. Čitaj: `RNN_with_prior_knowledge/README.md` (Knowledge-Guided)
3. Čitaj: `shared/docs/` (sva dokumentacija)
4. Počni treniranje

---

## 📈 OČEKIVANA NAPRETKA

```
┌─────────────────────────────────────────────┐
│          RMSE PROGRESSION                   │
├─────────────────────────────────────────────┤
│ RNN:                    688 W ❌            │
│ RNN1:                   232 W ✅ (66% ↓)    │
│ RNN2:                   ~180 W ✅✅ (23% ↓) │
│ RNN3-6:                 TBD 🔬             │
│                                             │
│ RNN_KGL_v1:             TBD 🧠             │
│ RNN_KGL_v2:             TBD 🧠             │
│ RNN_KGL_v3:             ~140-150 W 🚀      │
└─────────────────────────────────────────────┘
```

---

## 🗂️ GDJE JE ŠTA

| Trebam | Gdje |
|--------|------|
| **Razumijevanje baseline RNN** | `einfache_RNN/README.md` |
| **Razumijevanje Knowledge-Guided Learning** | `RNN_with_prior_knowledge/README.md` |
| **Specifikacije RNN modela** | `einfache_RNN/RNN*/README.md` |
| **Treniranje RNN modela** | `einfache_RNN/RNN*/einfacheRNN*_Colab.py` |
| **Sve hiperparametre detaljno** | `shared/docs/TRAINING_LOG.md` |
| **Usporedbe između modela** | `shared/docs/MODEL_GUIDE_DE.md` |
| **Testiranje svih modela** | `shared/tools/test_model.py` |
| **Visualizacija rezultata** | `shared/tools/plot_modeli.py` |
| **Originalni research paper** | `shared/references/Knowledge-Guided Learning PDF` |

---

## ✨ KLJUČNE DATOTEKE

### **Baseline (einfache_RNN)**
- 7 RNN modela (RNN do RNN6)
- Svaki ima: README.md, einfacheRNN*_Colab.py, models/, results/
- Svaki testira drukčiti hyperparameter

### **Knowledge-Guided (RNN_with_prior_knowledge)**
- 3 verzije KGL modela (v1, v2, v3)
- Svaki ugrađuje prior znanje o turbinama
- Svaki koristi Permissible States ograničenja

### **Shared Resources**
- Dokumentacija (5 .md datoteka)
- Test skripti (test_model.py, plot_modeli.py)
- Research paper (PDF)

---

## ❓ ČESTE PITANJA

**P: Gdje početi ako sam novi na projektu?**
O: Čitaj INDEX.md (sad) → einfache_RNN/README.md → RNN*/README.md

**P: Koji model trebam trenirati?**
O: Za brzi start: RNN2. Za eksperimente: RNN3-6. Za Knowledge: RNN_KGL_v1

**P: Koja je razlika između einfache_RNN i RNN_with_prior_knowledge?**
O: **einfache_RNN** = samo data-driven learning
   **RNN_with_prior_knowledge** = kombinacija data + domain knowledge

**P: Gdje je kod za Knowledge-Guided Learning?**
O: U RNN_with_prior_knowledge/ (još se kreira)

---

## 🎓 EDUKACIJSKI REDOSLIJED

1. **Razumijevanje baseline problema** (einfache_RNN/RNN)
   - Overfitting s 1 datasetom
   
2. **Multi-set treniranje** (einfache_RNN/RNN1)
   - Snaga datavarijabitilnosti
   
3. **Adaptive Learning Rate** (einfache_RNN/RNN2)
   - Scheduler za fine-tuning
   
4. **Hyperparameter eksperimenti** (einfache_RNN/RNN3-6)
   - LR spektrum, dubina mreže
   
5. **Knowledge-Guided Learning** (RNN_with_prior_knowledge)
   - Ugrađivanje prior znanja
   - Permissible States ograničenja

---

## 📅 VREMENSKE CRTE

| Aktivnost | Vrijeme |
|----------|---------|
| Setup (čitanje) | 20-30 min |
| Treniranje RNN2 | 40-50 min |
| Treniranje RNN3-6 | 160-200 min |
| Testing svih | 10-15 min |
| Knowledge-Guided (v1) | 60-90 min |
| **TOTAL** | **~5-6 sati** |

---

**INDEX Kreirano:** 2026-07-01  
**Zadnja Ažuriranja:** Reorganizacija u 2 glavna foldera  
**Verzija:** 3.0 (Categorized Structure)

---

## 🚀 READY?

**Faza 1 - Baseline:** `einfache_RNN/README.md` 🤖  
**Faza 2 - Knowledge:** `RNN_with_prior_knowledge/README.md` 🧠

**Let's go! 🎉**

