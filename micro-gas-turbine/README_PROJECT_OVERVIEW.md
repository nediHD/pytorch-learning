# 🚀 GAS TURBINE RNN PROJEKT - Kompletan Pregled

**Sve što trebašznati o projektu, RNN-u i što smo učili**

---

## 📌 PROJEKT: ŠTA RADI?

**Cilj:** Predvidjeti **snagu gas turbine** (P(t+1)) koristeći **sekvencu historijskih napetosti**.

```
Ulaz:  451 vremenski korak napetosti
       [V(t-450), V(t-449), ..., V(t-1), V(t)]
       
Obrada: RNN neuronska mreža
        (LSTM varijanta za duge sekvence)
        
Izlaz: Snaga turbine u sljedećem koraku
       P(t+1) = ? [Watts]
```

---

## 🧠 KAKO RNN FUNKCIONIRA?

### 1. **Što je RNN?**

RNN (Recurrent Neural Network) je neuronska mreža sa **petljama** - može pamtiti prethodna stanja!

```
Razlika od običnog neuralnog mreža (ANN):

ANN:  Ulaz → Sloj 1 → Sloj 2 → Izlaz
      (Nema memorije, svaki ulaz nezavisan)

RNN:  Ulaz[t] → RNN Cell → Izlaz[t]
              ↑_________↓
              (Memorija! h[t] prenosi info dalje)
      
      Ulaz[t+1] + h[t] → RNN Cell → Izlaz[t+1]
```

### 2. **Hidden State (Memorija)**

Ključna razlika: RNN ima **hidden state** `h[t]` koji se prenosi kroz vremenske korake.

```
Matematika:
  h[t] = activation( W_h * h[t-1] + W_x * X[t] )
  
Gdje:
  h[t]     = MEMORIJA u trenutku t
  h[t-1]   = Memorija iz prošlog koraka (što je RNN "zapamtio")
  X[t]     = Novi ulaz (napetost sada)
  W_h, W_x = Naučene težine
  
Što se dešava?
  1. Novi ulaz V(t) dolazi u RNN
  2. RNN koristi prethodnu memoriju h(t-1)
  3. Kombinira ih: h(t) = f(h(t-1), V(t))
  4. Memorija se prosleđuje dalje!
```

### 3. **Primjer: Kako RNN predviđa snagu**

```
SEKVENCA ULAZA (451 vremenski korak):
  t=1:   V[1]=4.2V  → h[1] = memorija1
  t=2:   V[2]=4.5V  → h[2] = memorija2 (koristi h[1]!)
  t=3:   V[3]=5.1V  → h[3] = memorija3 (koristi h[2]!)
  ...
  t=451: V[451]=6.8V → h[451] = memorija451
  
IZLAZ:
  P[452] = Dense(h[451]) = 1850W
  
Što je RNN naučio?
  - Trend: Jesu li napetosti rasle ili padale?
  - Brzina: Koliko brzo se mijenja?
  - Uzorak: Je li bilo koji periodičan uzorak?
  - Sve to koristi za predviđanje!
```

---

## 🏗️ ARHITEKTURA NAŠEG MODELA

### Naša struktura (RNN2-RNN6):

```
INPUT (451 vremenski korak)
  ↓
LSTM Sloj 1 (32 neurona) → h[1]
  ↓
LSTM Sloj 2 (32 neurona) → h[2]  
  ↓
LSTM Sloj 3 (32 neurona) → h[3]  (ili 4 sloja kod RNN4, 2 sloja kod RNN5)
  ↓
Uzmi zadnji hidden state h[451]
  ↓
Dense sloj (1 neuron)
  ↓
OUTPUT: Predviđena snaga P(t+1)
```

### Zašto LSTM umjesto običnog RNN?

```
Problem običnog RNN-a:
  - Vanishing gradient - dugačke sekvence (451 koraka!) = loše učenje
  
LSTM rješenje:
  - Ima "cell state" (dugoročna memorija)
  - Ima "kapije" (forget gate, input gate, output gate)
  - Može pamtiti važne stvari kroz 451 koraka!
  
Matematika LSTM:
  f[t] = forget gate (što zaboraviti?)
  i[t] = input gate (što zapamtiti?)
  c[t] = cell state (dugoročna memorija)
  o[t] = output gate (što pokazati?)
  
  c[t] = f[t] * c[t-1] + i[t] * c̃[t]  ← DUGOROČNA MEMORIJA!
  h[t] = o[t] * tanh(c[t])
```

---

## 📊 PROGRESIJA MODELA - ŠTO SMO UČILI

### Faza 1: **RNN (Baseline)**
```
Što: Treniranje samo na 1 datasetu (ex_1.csv)
Rezultat: RMSE = 688 W ❌
Lekcija: Jedan dataset = OVERFITTING
         Model ne generalizira na druge datasete
```

### Faza 2: **RNN1 (Multi-Set)**
```
Što: Treniranje na 6 datasetā (ex_1, ex_9, ex_20, ex_21, ex_23, ex_24)
Podatci: 9,469 → 53,754 sekvencí (5.7x više!)
Rezultat: RMSE = 232 W ✅ (66% BOLJE!)
Lekcija: Datavarijabitilnost je kritična!
         Više različitih primjera = bolja generalizacija
```

### Faza 3: **RNN2 (Scheduler)**
```
Što: RNN1 + Learning Rate Scheduler
Scheduler: ReduceLROnPlateau (smanjuje LR kada stagna)

Kako radi:
  Epoche 1-10:   LR = 0.001 → Loss pada brzo
  Epoche 11-20:  LR = 0.001 → Loss stagnira
                 ↓ SCHEDULER AKTIVIRAN!
  Epoche 21:     LR = 0.0005 → Loss pada opet! (fine-tuning)
  
Rezultat: RMSE = ~180 W ✅✅ (23% dodatno bolje!)
Lekcija: Adaptivni learning rate → bolje fine-tuning
         Ne trebam fiksni LR tijekom treniranja
```

### Faza 4: **RNN3-RNN6 (Hyperparameter eksperimenti)**

```
RNN3: Viši početni LR (0.01 umjesto 0.001)
      ↳ Testira: Može li brže učenje biti bolje?

RNN4: 4 LSTM sloja umjesto 3
      ↳ Testira: Trebam li više kapaciteta (dublja mreža)?

RNN5: 2 LSTM sloja umjesto 3
      ↳ Testira: Mogu li sa manje parametara?

RNN6: Niži početni LR (0.0005 umjesto 0.001)
      ↳ Testira: Je li precizniji početak bolji?
      
Rezultati: TBD 🔬 (trebaju treniranje u Colab-u)
```

---

## 🎯 KLJUČNE KONCEPTE KOJE SU NAUČENI

### 1. **Overfitting i Generalizacija**
```
Overfitting: Model dobro radi na train data ali loše na test data
             (RNN primjer: samo ex_1 → RMSE 688W na ex_22)

Rješenja:
  ✅ Više trining podataka (više datasetā) → RNN1
  ✅ Regulizacija (dropout, L1/L2) - nismo implementirali
  ✅ Early Stopping - zaustavljamo prije nego što se pretreniram
  ✅ Cross-validation - testiramo na različitim setovima
```

### 2. **Learning Rate**
```
Što je LR?
  Parametar koji kontrolira koliko "brzo" model uči
  
Premali LR: Učenje je SPOROOO, trebat će vječnost
Previsoki LR: Učenje je nestabilno, loss oscilira

Idealno: Počnemo s većim LR (brže), pa polako smanjujemo (fine-tuning)
        ↳ To je što radi ReduceLROnPlateau scheduler!
```

### 3. **Early Stopping**
```
Što je to?
  Zaustavimo treniranje kada validation loss prestane padati

Zaštita od:
  - Overfittinga (treniranje se nastavlja, ali model se gore!)
  - Vrtnjanja vremena (ako nema poboljšanja, zašto nastaviti?)

U našim modelima:
  patience=20 znači: Ako nema poboljšanja 20 epoha → STOP
```

### 4. **Batch Processing**
```
Umjesto jednog primjera po koraku:
  Batch size = 32
  ↓
  Obradimo 32 primjera odjednom
  ↓
  Ažuriramo težine
  
Prednosti:
  ✅ Brže (batch operacije su paralelne)
  ✅ Stabilnije (više primjera = jasniji gradient)
  
U našem slučaju:
  53,754 sekvencí / 32 batch_size = 1,679 batchā po epohi
```

### 5. **Normalizacija**
```
Problem: Podaci V, P na različitim skalama
         V = [3-10] V, P = [932-3249] W
         
Rješenje: Normalizacija na [0, 1]
  X_norm = (X - X_min) / (X_max - X_min)
  
Razlog: Neural mreža bolje radi sa vrijednostima [0, 1]
        Izbjeći: Eksplozija gradijenata, loša konvergencija
```

---

## 🔬 EKSPERIMENTALNI PROCES

### Što smo činili?

```
1. PROBLEM: RNN s 1 datasetom → loši rezultati (688W)
2. HIPOTEZA: Trebam više različitih podataka
3. EKSPERIMENT: Multi-set (6 datasetā)
4. REZULTAT: 232W (66% bolje!) ✅
5. ZAKLJUČAK: Datavarijabitilnost kritična

PONAVLJANJE:
1. PROBLEM: RNN1 s fiksnim LR → platau u treniranju
2. HIPOTEZA: Trebam adaptivni LR
3. EKSPERIMENT: Scheduler
4. REZULTAT: 180W (23% bolje!) ✅
5. ZAKLJUČAK: Scheduler pomaže fine-tuning
```

### Što trebamo dalje?

```
Hipotetski rezultati RNN3-6:
  
  RNN3 (0.01 LR):    ? (možda bolje, ali rizik oscilacija)
  RNN4 (4 sloja):    ? (više kapaciteta, ali sporije treniranje)
  RNN5 (2 sloja):    ? (manje parametara, brže, ali dovoljno?)
  RNN6 (0.0005 LR):  ? (precizniji početak, ali sporiji)
  
Trebamo trenirati i vidjeti šta će biti najbolje!
```

---

## 🧪 KAKO TRENIRATI (BRZI PREGLED)

```
1. PRIPREMA:
   ✅ Kod je u einfache_RNN/RNN2/einfacheRNN2_Colab.py
   ✅ Dokumentacija u shared/docs/TRAINING_LOG.md
   
2. GOOGLE COLAB:
   - Otvori https://colab.research.google.com/
   - Runtime → Change runtime type → GPU
   - Kopira kod iz einfacheRNN*_Colab.py
   - Zalijepi u Colab cell, Shift+Enter
   - Učitaj 6 CSV datoteka: ex_1, ex_9, ex_20, ex_21, ex_23, ex_24
   - Čekaj ~40-50 minuta
   
3. REZULTATI:
   - Preuzmi einfachesRNN2.pth (model weights)
   - Spremi u einfache_RNN/RNN2/models/
   
4. EVALUACIJA:
   - python shared/tools/test_model.py
   - python shared/tools/plot_modeli.py
```

---

## 📚 ZNANJA KOJU TREBAM ZA RAZGOVOR

**Minimalno (Fast Track):**
```
1. ŠTA JE RNN?
   → Neuronska mreža sa memorijom (hidden state)
   
2. ZAŠTO RNN ZA GAS TURBINE?
   → Sekvencijalni ulazi, trebam pamtiti historiju
   
3. ŠTA JE LSTM?
   → RNN varijanta sa dugoročnom memorijom (cell state)
   
4. NAŠI MODELI?
   → RNN (baseline) → RNN1 (multi-set) → RNN2 (scheduler) → RNN3-6 (eksperimenti)
   
5. REZULTATI?
   → 688W → 232W → ~180W (progresivno poboljšanje)
```

**Detaljnije (Deep Dive):**
```
- Kako funkcionira hidden state (h[t] formula)
- Learning Rate Scheduler (ReduceLROnPlateau)
- Early Stopping i overfitting
- Batch processing
- Normalizacija
- Multi-dataset training
- LSTM arhitektura vs. obični RNN
```

**Prednji plan:**
```
- Zašto je datavarijabitilnost bitna
- Zašto je adaptivni LR bolji od fiksnog
- Kako se testira model (RMSE, MAE, MSE)
- Što su RNN3-6 eksperimenti testirali
```

---

## 💡 PITANJA KOJU MOŽEŠ OČEKIVATI

```
1. "Što je razlika između RNN i ANN?"
   → ANN nema memorije, RNN ima hidden state

2. "Zašto koristiš multi-dataset treniranje?"
   → Izbjeći overfitting, poboljšati generalizaciju

3. "Što radi Learning Rate Scheduler?"
   → Smanjuje LR kada stagna, omogućava fine-tuning

4. "Gdje je LSTM u tvojim modelima?"
   → Korištena je LSTM varijanta RNN-a (nn.LSTM u PyTorch-u)

5. "Kako testiramo model?"
   → RMSE na test datasetima (ex_4.csv, ex_22.csv)

6. "Što su RNN3-6 eksperimenti?"
   → Test različitih hyperparametara (LR, dubina mreže)

7. "Koji je model najbolji?"
   → Još trebamo trenirati RNN3-6 i vidjeti rezultate
```

---

## 🎓 BRZI KOMENTAR ZA RAZGOVOR

```
"Projektam koristi LSTM varijantom RNN-a da predviđa snagu gas turbine
na temelju sekvence od 451 vremenskog koraka napetosti.

Počeli smo s baseline RNN-om (samo 1 dataset → 688W RMSE).
Zatim smo koristili multi-dataset treniranje (6 datasets → 232W RMSE).
Dodali smo Learning Rate Scheduler za fine-tuning (→ ~180W RMSE).

Sada testiramo različite hyperparametre (RNN3-6) da vidimo što još
može poboljšati performanse prije nego što dodamo Knowledge-Guided
Learning s Permissible States ograničenjima."
```

---

## 📖 KORISNI LINKOVI U PROJEKTU

- `einfache_RNN/README.md` - Vodiči za baseline modele
- `RNN_with_prior_knowledge/README.md` - Knowledge-Guided Learning
- `shared/docs/TRAINING_LOG.md` - Svi hiperparametri detalje
- `shared/docs/ANN_vs_RNN_DE.md` - Razlika između ANN i RNN
- `shared/docs/MODEL_GUIDE_DE.md` - Usporedbe između modela
- `INDEX.md` - Master navigation

---

**Kreirano:** 2026-07-01  
**Svrha:** Priprema za razgovor i sumarni pregled projekta  
**Zadnja ažuriranja:** Sve što smo učili kroz vježbe

