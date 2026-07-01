# 🤖 EINFACHE RNN - Baseline Modeli

---

## 📋 ŠTO SE NALAZI OVDJE

Ovo je folder koji sadrži sve **osnovne RNN modele bez dodatnih ograničenja**.

Modeli su organizirani kao progresija poboljšanja od baseline-a do najpreciznijeg modela.

---

## 📂 STRUKTURA

```
einfache_RNN/
├── RNN/           (1 dataset - Baseline, RMSE 688W)
├── RNN1/          (6 datasets - Multi-set, RMSE 232W)
├── RNN2/          (6 datasets + Scheduler - RMSE ~180W)
├── RNN3/          (6 datasets + Scheduler + Viši LR)
├── RNN4/          (6 datasets + Scheduler + 4 LSTM sloja)
├── RNN5/          (6 datasets + Scheduler + 2 LSTM sloja)
└── RNN6/          (6 datasets + Scheduler + Niži LR)
```

---

## 🎯 SVRHA

Ova serija modela pokazuje:

1. **RNN** - Baseline greške (overfitting na 1 dataset)
2. **RNN1** - Multi-set treniranje + poboljšanja (66% bolje!)
3. **RNN2** - Learning Rate Scheduler (23% dalje bolje)
4. **RNN3-RNN6** - Hyperparameter eksperimenti

Cilj je razumijevanje što pomaže poboljšati RMSE prije nego što dodamo **Knowledge-Guided Learning** (prior knowledge) ograničenja.

---

## 🚀 KAKO POČETI

1. Čitaj `../INDEX.md` za master vodiče
2. Odaberi model: `RNN*/README.md`
3. Treniraj u Google Colab
4. Testiraj sa `../shared/tools/test_model.py`

---

## 📊 OČEKIVANI REZULTATI

| Model | RMSE ex_22 | Status |
|-------|-----------|--------|
| RNN | 688 W | ❌ |
| RNN1 | 232 W | ✅ |
| RNN2 | ~180 W | ✅✅ |
| RNN3-6 | TBD | 🔬 |

---

**Status:** Baseline eksperimenti - Gotov za treniranje

