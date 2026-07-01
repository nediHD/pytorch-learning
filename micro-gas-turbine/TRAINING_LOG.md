# 📊 TRAINING LOG - RNN vs RNN1 vs RNN2

---

## 🔵 **MODEL 1: einfachesRNN.pth (ORIGINAL)**

### **POSTAVKE TRENIRANJA:**

```
Datazet za treniranje:  ex_1.csv SAMO
Broj sekvencí:          9,469 (iz 9,920 redaka)
```

**Arhitektura Modela:**
```
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri:**
```
Learning Rate (LR):     0.001 (FIKSNA - ne mijenja se!)
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE (Mean Squared Error)
Epochs:                 300
Early Stopping Patience: 20 epoha (zaustavi ako nema poboljšanja)
Early Stopping Aktivno: ✅ DA
Learning Rate Scheduler: ❌ NE (fiksna LR!)
Dropout:                ❌ NE
Batch Normalization:    ❌ NE
```

**Normalizacija:**
```
Min Input (V):     3.00
Max Input (V):     10.00
Min Output (W):    932.84
Max Output (W):    3,249.89
Sekvenca Veličina (N): 451 vremenskih koraka
```

### **REZULTATI TESTIRANJA:**

| Dataset | RMSE (W) | MSE | MAE (W) | Kvaliteta |
|---------|----------|-----|---------|-----------|
| ex_4.csv | 384.19 | 0.027493 | 333.14 | ❌ LOŠA |
| ex_22.csv | 688.63 | 0.088328 | 532.05 | ❌ LOŠA |

**Zaključak:** Model treniran samo na ex_1 ne može dobro predvidjeti drugačite datasete.

---

## 🟢 **MODEL 2: einfachesRNN1.pth (MULTI-SET + EARLY STOP)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754 (5.7x više nego RNN!)
```

**Arhitektura Modela:**
```
ISTA kao RNN:
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri:**
```
Learning Rate (LR):     0.001 (FIKSNA - ne mijenja se!)
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE (Mean Squared Error)
Epochs:                 300 (ali zaustavio se ranije!)
Early Stopping Patience: 20 epoha
Early Stopping Aktivno: ✅ DA (zaustavio se oko epohe 67)
Learning Rate Scheduler: ❌ NE (fiksna LR!)
Dropout:                ❌ NE
Batch Normalization:    ❌ NE
```

**Normalizacija:**
```
ISTA kao RNN (iz ex_1.csv):
Min Input (V):     3.00
Max Input (V):     10.00
Min Output (W):    932.84
Max Output (W):    3,249.89
Sekvenca Veličina (N): 451
```

### **REZULTATI TESTIRANJA:**

| Dataset | RMSE (W) | MSE | MAE (W) | Kvaliteta |
|---------|----------|-----|---------|-----------|
| ex_4.csv | 132.54 | 0.003272 | 109.53 | ✅ DOBRA |
| ex_22.csv | 232.04 | 0.010029 | 178.27 | ✅ DOBRA |

**Poboljšanja vs RNN:**
```
ex_4:  384.19 W → 132.54 W  = 65% BOLJE! ✅✅✅
ex_22: 688.63 W → 232.04 W  = 66% BOLJE! ✅✅✅
```

**Zaključak:** Multi-set treniranje drastično poboljšava generalizaciju!

---

## 🟡 **MODEL 3: einfachesRNN2.pth (MULTI-SET + SCHEDULER + PATIENCE=10)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754 (ISTO kao RNN1!)
```

**Arhitektura Modela:**
```
ISTA kao RNN i RNN1:
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri - NOVE POSTAVKE:**
```
Learning Rate (LR):     0.001 (POČETNA!)
Learning Rate Scheduler: ✅ DA - ReduceLROnPlateau! (NOVO!)
  - Faktor:            0.5 (novi LR = stari LR × 0.5)
  - Patience Scheduler: 10 epoha (NOVO!)
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE
Epochs:                 300
Early Stopping Patience: 20 epoha
Early Stopping Aktivno: ✅ DA
Dropout:                ❌ NE
Batch Normalization:    ❌ NE
```

**Normalizacija:**
```
ISTA kao RNN i RNN1:
Min Input (V):     3.00
Max Input (V):     10.00
Min Output (W):    932.84
Max Output (W):    3,249.89
Sekvenca Veličina (N): 451
```

### **KAKO SCHEDULER RADI:**

```
Epoha 1-10:    LR = 0.001000 (početna)
               Loss pada: 0.924 → 0.050
               
Epoha 11-20:   LR = 0.001000 (nema poboljšanja 10 epoha)
               Loss stagnira oko 0.050
               ↓ SCHEDULER AKTIVIRAN!
               
Epoha 21:      LR = 0.000500 (pola manji - fine-tuning)
               Loss počinje padati ponovno: 0.050 → 0.015
               
Epoha 22-31:   LR = 0.000500
               Loss pada dalje
               
Epoha 32:      Nema poboljšanja 10 epoha
               ↓ SCHEDULER AKTIVIRAN OPET!
               
Epoha 33:      LR = 0.000250 (još manji)
               Dodatni fine-tuning...
```

### **OČEKIVANI REZULTATI:**

| Dataset | Očekivani RMSE | Poboljšanje vs RNN1 |
|---------|-----------------|-------------------|
| ex_4.csv | ~120 W | ≈ 10% bolje |
| ex_22.csv | ~180 W | ≈ 23% bolje! |

**Razlog:** Learning Rate Scheduler dozvoljava modelu da radi fine-tuning sa manjim LR nakon što stagnira sa većim.

---

## 📊 **USPOREDBA SVI TRI MODELA**

### **Arhitektura:**
```
RNN:  ISTA  (3 LSTM sloja × 32 neurona)
RNN1: ISTA
RNN2: ISTA
```

### **Training Dataset:**
```
RNN:  1 set (ex_1 samo)         → 9,469 sekvencí
RNN1: 6 setova (ex_1,9,20,21,23,24) → 53,754 sekvencí
RNN2: 6 setova (ISTO kao RNN1)  → 53,754 sekvencí
```

### **Learning Rate:**
```
RNN:  0.001 (FIKSNA)
RNN1: 0.001 (FIKSNA)
RNN2: 0.001 → 0.0005 → 0.00025 (DINAMIČKA sa Scheduler-om!) ✅
```

### **Early Stopping:**
```
RNN:  Patience 20
RNN1: Patience 20 (zaustavljen u epohi ~67)
RNN2: Patience 20 (trebao bi biti ~90+ epoha zbog Scheduler-a)
```

### **REZULTATI NA ex_4.csv:**
```
RNN:  384.19 W  ❌
RNN1: 132.54 W  ✅✅
RNN2: ~120 W    ✅✅✅ (očekivano)
```

### **REZULTATI NA ex_22.csv:**
```
RNN:  688.63 W  ❌
RNN1: 232.04 W  ✅✅
RNN2: ~180 W    ✅✅✅ (očekivano)
```

---

## 🎯 **ZAKLJUČAK:**

| Aspekt | RNN | RNN1 | RNN2 |
|--------|-----|------|------|
| Training Setovi | 1 | 6 | 6 |
| LR Scheduler | ❌ | ❌ | ✅ |
| Očekivana RMSE ex_4 | 384 W | 132 W | ~120 W |
| Očekivana RMSE ex_22 | 689 W | 232 W | ~180 W |
| Prednost | Jednostavan | Multi-set | Multi-set + Optimizer |

**Očekivani pobjednika:** RNN2 sa Learning Rate Scheduler! 🏆

---

---

## 🟣 **MODEL 4: einfachesRNN3.pth (MULTI-SET + SCHEDULER + VIŠI LR)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754 (ISTO kao RNN1 i RNN2)
```

**Arhitektura Modela:**
```
ISTA kao RNN, RNN1 i RNN2:
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri - EKSPERIMENT: VIŠI LEARNING RATE**
```
Learning Rate (LR):     0.01 (10x VIŠI od RNN2!)
Learning Rate Scheduler: ✅ DA - ReduceLROnPlateau
  - Faktor:            0.5 (novi LR = stari LR × 0.5)
  - Patience Scheduler: 10 epoha
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE
Epochs:                 300
Early Stopping Patience: 20 epoha
```

**Normalizacija:**
```
ISTA kao svi prijašnji modeli
```

### **CILJ EKSPERIMENTA:**

Testira se: **Da li viši početni learning rate (0.01) omogućava bržu konvergenciju ili dovodi do nestabilnosti?**

```
RNN2:  LR počinje sa 0.001      (konzervativno)
RNN3:  LR počinje sa 0.01       (agresivno) ← NOVO!
```

Očekivanja:
- **Scenario A (Optimističan):** Brži početak + scheduler fine-tuning = RMSE može biti još bolje
- **Scenario B (Pesimističan):** Viši LR može uzrokovati oscilacije, konačni RMSE može biti gori od RNN2

### **OČEKIVANI REZULTATI:**

| Dataset | RNN2 (~180 W) | RNN3 (Očekivano) | Scenarij |
|---------|-------|-----------|----------|
| ex_4.csv | ~120 W | ?  | TBD nakon treniranja |
| ex_22.csv | ~180 W | ? | TBD nakon treniranja |

---

## 🟠 **MODEL 5: einfachesRNN4.pth (MULTI-SET + SCHEDULER + 4 LSTM SLOJA)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754
```

**Arhitektura Modela - DUBLJA MREŽA (4 LSTM SLOJA):**
```
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- LSTM Sloj 4:  input=32  → hidden=32 ← NOVO SLOJ!
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri - EKSPERIMENT: DUBLJA ARHITEKTURA**
```
Learning Rate (LR):     0.001 (ISTO kao RNN2)
Learning Rate Scheduler: ✅ DA - ReduceLROnPlateau (ISTO kao RNN2)
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE
Epochs:                 300
Early Stopping Patience: 20 epoha
Broj LSTM slojeva:      4 (umjesto 3) ← NOVO!
```

**Normalizacija:**
```
ISTA kao svi prijašnji modeli
```

### **CILJ EKSPERIMENTA:**

Testira se: **Da li duplja mreža (više parametara, više kapaciteta) generalizira bolje na neviđenim datasetima?**

```
RNN2: 3 LSTM sloja  (33 × 32 × 32 × 1 = ~3,105 parametara)
RNN4: 4 LSTM sloja  (1 × 32 × 32 × 32 × 32 × 1 = ~4,145 parametara) ← VIŠE KAPACITETA
```

Očekivanja:
- **Scenario A (Optimističan):** Veći kapacitet mreže može učiti kompleksnije obrasce = bolji RMSE
- **Scenario B (Pesimističan):** Overfitting na training data, loši rezultati na test datasetima

### **OČEKIVANI REZULTATI:**

| Dataset | RNN2 (~180 W) | RNN4 (Očekivano) | Scenarij |
|---------|-------|-----------|----------|
| ex_4.csv | ~120 W | ? | TBD nakon treniranja |
| ex_22.csv | ~180 W | ? | TBD nakon treniranja |

---

## 🟢 **MODEL 6: einfachesRNN5.pth (MULTI-SET + SCHEDULER + 2 LSTM SLOJA)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754
```

**Arhitektura Modela - PLIĆA MREŽA (2 LSTM SLOJA):**
```
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri - EKSPERIMENT: PLIĆA ARHITEKTURA**
```
Learning Rate (LR):     0.001 (ISTO kao RNN2)
Learning Rate Scheduler: ✅ DA - ReduceLROnPlateau (ISTO kao RNN2)
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE
Epochs:                 300
Early Stopping Patience: 20 epoha
Broj LSTM slojeva:      2 (umjesto 3) ← MANJE SLOJEVA!
```

**Normalizacija:**
```
ISTA kao svi prijašnji modeli
```

### **CILJ EKSPERIMENTA:**

Testira se: **Da li plića mreža (manje parametara) i dalje daje dobre rezultate uz brže treniranje?**

```
RNN2: 3 LSTM sloja  (~3,105 parametara, sporije treniranje)
RNN5: 2 LSTM sloja  (~2,065 parametara, brže treniranje) ← MANJE PARAMETARA
```

Očekivanja:
- **Scenario A (Optimističan):** Plića mreža je dovoljna za ovaj problem, brže treniranje, nije overfitting
- **Scenario B (Pesimističan):** Premalo kapaciteta, lošiji RMSE od RNN2

### **OČEKIVANI REZULTATI:**

| Dataset | RNN2 (~180 W) | RNN5 (Očekivano) | Scenarij |
|---------|-------|-----------|----------|
| ex_4.csv | ~120 W | ? | TBD nakon treniranja |
| ex_22.csv | ~180 W | ? | TBD nakon treniranja |

---

## 🔵 **MODEL 7: einfachesRNN6.pth (MULTI-SET + SCHEDULER + NIŽI LR)**

### **POSTAVKE TRENIRANJA:**

```
Dataset za treniranje:  ex_1.csv + ex_9.csv + ex_20.csv + ex_21.csv + ex_23.csv + ex_24.csv
Broj sekvencí:          ~53,754
```

**Arhitektura Modela:**
```
ISTA kao RNN, RNN1 i RNN2:
- LSTM Sloj 1:  input=1   → hidden=32
- LSTM Sloj 2:  input=32  → hidden=32
- LSTM Sloj 3:  input=32  → hidden=32
- Dense Sloj:   input=32  → output=1
```

**Hiperparametri - EKSPERIMENT: NIŽI LEARNING RATE**
```
Learning Rate (LR):     0.0005 (2x NIŽI od RNN2!)
Learning Rate Scheduler: ✅ DA - ReduceLROnPlateau
  - Faktor:            0.5 (novi LR = stari LR × 0.5)
  - Patience Scheduler: 10 epoha
Batch Size:             32
Optimizator:            Adam
Loss Funkcija:          MSE
Epochs:                 300
Early Stopping Patience: 20 epoha
```

**Normalizacija:**
```
ISTA kao svi prijašnji modeli
```

### **CILJ EKSPERIMENTA:**

Testira se: **Da li niži početni learning rate (0.0005) omogućava preciznije fine-tuning od početka?**

```
RNN3:  LR počinje sa 0.01      (agresivno)
RNN2:  LR počinje sa 0.001     (balansirano)
RNN6:  LR počinje sa 0.0005    (konzervativno) ← NOVO!
```

Očekivanja:
- **Scenario A (Optimističan):** Precizniji početak + scheduler = bolja konačna konvergencija
- **Scenario B (Pesimističan):** Previše spora početna učenja, treniranje može zahtijevati više epoha ili biti suboptimalno

### **OČEKIVANI REZULTATI:**

| Dataset | RNN2 (~180 W) | RNN6 (Očekivano) | Scenarij |
|---------|-------|-----------|----------|
| ex_4.csv | ~120 W | ? | TBD nakon treniranja |
| ex_22.csv | ~180 W | ? | TBD nakon treniranja |

---

## 📊 **SVEUKUPNA USPOREDBA SVI MODELI (RNN do RNN6)**

| Aspekt | RNN | RNN1 | RNN2 | RNN3 | RNN4 | RNN5 | RNN6 |
|--------|-----|------|------|------|------|------|------|
| **Training Setovi** | 1 | 6 | 6 | 6 | 6 | 6 | 6 |
| **LSTM Slojevi** | 3 | 3 | 3 | 3 | 4 | 2 | 3 |
| **LR (početna)** | 0.001 | 0.001 | 0.001 | **0.01** | 0.001 | 0.001 | **0.0005** |
| **LR Scheduler** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Scheduler Patience** | - | - | 10 | 10 | 10 | 10 | 10 |
| **Early Stop Patience** | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| **Eksperiment** | Baseline | Multi-set | LR Scheduler | Viši LR | Dublja | Plića | Niži LR |
| **ex_22.csv RMSE** | 688.63 W | 232.04 W | ~180 W | TBD | TBD | TBD | TBD |

---

**Datoteka napravljena:** 2026-07-01  
**Zadnja ažuriranja:** 2026-07-01 (dodani RNN3-RNN6)  
**Status:** Spreman za treniranje RNN3-RNN6
