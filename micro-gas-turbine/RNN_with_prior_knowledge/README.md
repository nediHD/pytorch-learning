# 🧠 RNN WITH PRIOR KNOWLEDGE - Knowledge-Guided Learning

---

## 📋 ŠTA SE NALAZI OVDJE

Ovo je folder za **Knowledge-Guided Learning** modele koji koriste **Permissible System States** ograničenja.

Ovi modeli ugrađuju fizikalno znanje o gas turbinama u neural network tijekom treniranja.

---

## 🎯 SVRHA

Umjesto da model samo nauči podatke, ovdje dodajemo:

- **Permissible States**: Samo validna stanja turbine
- **Physical Constraints**: Fizikalna znanja o turbinama
- **Prior Knowledge**: Inženjerskо znanje ugrađeno u model

---

## 📚 TEORIJA

Iz papera "Knowledge-Guided Learning of Temporal Dynamics..."

```
Cilj: Kombinirati data-driven learning s prior domain knowledge

Pristup:
  1. Učiti RNN s multi-set podacima (kao RNN1-6)
  2. Dodati loss funkciju koja penalizira nepermissible stanja
  3. Koristiti Permissible States kao ograničenja tijekom treniranja
```

---

## 📂 STRUKTURA (Će biti popunjena)

```
RNN_with_prior_knowledge/
├── RNN_KGL_v1/           (Knowledge-Guided RNN v1)
├── RNN_KGL_v2/           (s Permissible States)
└── RNN_KGL_v3/           (Optimizovani verzija)
```

---

## 🚀 NADALJE PLANOVI

1. **RNN_KGL_v1**: Bazna Knowledge-Guided verzija
   - Koristi RNN2 kao baseline
   - Dodaj Permissible States loss funkciju
   - Testiraj na ex_4.csv i ex_22.csv

2. **RNN_KGL_v2**: Optimizovano sa prior knowledge
   - Fine-tune Permissible States parametri
   - Kombinacija data-driven + knowledge-driven loss-a

3. **RNN_KGL_v3**: Finalna verzija
   - Best hyperparameters
   - Maksimalna preciznost + validnost

---

## 📊 OČEKIVANI REZULTATI

```
einfache_RNN/RNN2:    ~180 W (bez znanja)
RNN_KGL_v1:           TBD (s osnovnim znanjem)
RNN_KGL_v2:           TBD (optimizovano)
RNN_KGL_v3:           ~140-150 W? (najbolje!)
```

---

## 📖 REFERENCE

- Paper: "Knowledge-Guided Learning of Temporal Dynamics and its Application to Gas Turbines"
- Location: `../shared/references/`

---

**Status:** Pripremanje za Knowledge-Guided Learning fazu

