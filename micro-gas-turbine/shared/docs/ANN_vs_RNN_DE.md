# 🧠 ANN vs RNN - Detaljno Objašnjenje

---

## 📌 BRZA USPOREDBA

| Aspekt | ANN | RNN |
|--------|-----|-----|
| **Arhitektura** | Feed-forward (jedan smjer) | Recurrent (petlje) |
| **Memorija** | ❌ Nema | ✅ Ima (hidden state) |
| **Ulaz** | Fiksna veličina | Varijabilna sekvenca |
| **Zadaci** | Klasifikacija, regresija | Sekvence, vremenske serije |
| **Brzina** | Brža | Sporija |
| **Za gas turbine** | ❌ Loše | ✅ Odličan izbor |

---

## 🔴 **ANN (Artificial Neural Network)**

### Što je to?

**ANN** je osnovna neuronska mreža gdje informacije teku samo **u jednom smjeru** - od ulaza prema izlazu.

```
ULAZ → Skriveni sloj 1 → Skriveni sloj 2 → IZLAZ
  ↓                ↓                    ↓
 [1,2,3]         [h1,h2]              [Y]
```

### Karakteristike

- ✅ Jednostavna za razumijevanje
- ✅ Brza obuka
- ✅ Dobra za klasifikaciju slike
- ❌ **NEMA MEMORIJE** između ulaza
- ❌ Ne razumije redoslijed ili vremenske ovisnosti
- ❌ Svaki ulaz tretira kao nezavisan

### Primjer Ulaza

```
Predviđanje cijene kuće:
  Ulaz:  [površina=100m², sobe=3, lokacija=5]
  Izlaz: Cijena=500,000 €

Svaki put kada šalješ input, ANN ga obradi kao potpuno novog.
Nema "sjećanja" o prethodnim ulazima.
```

### Matematika

```
Ulaz x → Linearni sloj → Aktivacijska funkcija → Skriveni sloj
                                                      ↓
                                              Ponovno linearni sloj
                                                      ↓
                                                    IZLAZ
```

**Problem:** Ako redoslijed promijeniš, izlaz je potpuno drugačiji (loše za vremenske serije!)

---

## 🔵 **RNN (Recurrent Neural Network)**

### Što je to?

**RNN** je neuronska mreža sa **petljama** - informacije se mogu kretati i unatrag!

Bitna razlika: RNN ima **hidden state** (skriveno stanje) koje se prenosi od trenutka k trenutku.

```
ULAZ[t] → SKRIVENI SLOJ[t] → IZLAZ[t]
            ↓              ↑
            └──────────────┘ (petlja - memorija!)
            
ULAZ[t+1] → SKRIVENI SLOJ[t+1] → IZLAZ[t+1]
             (koristi SLOJ[t]!)
```

### Karakteristike

- ✅ **IMA MEMORIJU** - sjeti se prethodnih stanja
- ✅ Odličan za vremenske serije
- ✅ Razumije redoslijed i kontekst
- ✅ Idealan za sekvencijalne podatke
- ❌ Sporija obuka
- ❌ Teža za razumijevanje i debug
- ❌ Problem nestajućih gradijenata (vanishing gradient)

### Primjer Ulaza

```
Predikcija snage turbine:
  Ulaz:  [V(t-2)=4.5V, V(t-1)=5.2V, V(t)=6.1V] ← SEKVENCA!
  
RNN će:
  1. Obraditi V(t-2), stvoriti hidden state h1
  2. Obraditi V(t-1), koristiti h1 + V(t-1), stvoriti h2
  3. Obraditi V(t), koristiti h2 + V(t), stvoriti h3
  4. Iz h3 predvidjeti: P(t+1) = 1850W

Sekvenca DOISTA BITNA! RNN pamti redoslijed.
```

### Matematika (Simplified)

```
h[t] = activation( W_h * h[t-1] + W_x * x[t] + b )
y[t] = W_y * h[t] + b_y

Gdje:
  h[t]   = skriveno stanje u trenutku t (MEMORIJA!)
  h[t-1] = prethodno skriveno stanje (iz prošlog vremena)
  x[t]   = ulaz u trenutku t
  y[t]   = izlaz u trenutku t
```

---

## 🎯 **VIZUALNA RAZLIKA**

### ANN Arhitektura (Fiksni ulaz)

```
        ┌─────────┐
        │ Ulaz 1  │
        └────┬────┘
             │
        ┌─────────┐      ┌──────────┐      ┌────────┐
        │ Ulaz 2  ├──────┤ Skriveni ├──────┤ Izlaz  │
        └─────────┘      │ sloj     │      └────────┘
        ┌─────────┐      └──────────┘
        │ Ulaz 3  │
        └─────────┘
        
Sve ide u jednom smjeru → (aciklički graf)
```

### RNN Arhitektura (Vremenske serije)

```
Vremenska sekvenca:

t=1: Ulaz[1] → [RNN Cell] → Izlaz[1]
                   ↓
                 h[1] (memorija)
                   ↓
t=2: Ulaz[2] + h[1] → [RNN Cell] → Izlaz[2]
                         ↓
                       h[2] (memorija)
                         ↓
t=3: Ulaz[3] + h[2] → [RNN Cell] → Izlaz[3]
                         ↓
                       h[3] (memorija)

Petlje! Memorija iz prošlog vremena → sada korištenja!
```

---

## 💡 **PRAKTIČNI PRIMJERI**

### ANN: Kada koristiti?

```
✅ Klasifikacija slike:
   Ulaz: Slika (784 piksela)
   Izlaz: Klasa (0-9 za brojeve)
   
✅ Jednostavna regresija:
   Ulaz: Osobine
   Izlaz: Cijena
   
❌ Predviđanje vremenske serije (LOŠE!)
   Ulaz: [temp(t)], izlaz: temp(t+1)
   Problem: Nema učenja sekvence!
```

### RNN: Kada koristiti?

```
✅ Predviđanje vremenske serije (ODLIČAN IZBOR!):
   Ulaz: [V(t-450)...V(t)]
   Izlaz: P(t+1) - snaga turbine
   
✅ Obrada teksta:
   Ulaz: [token1, token2, token3...]
   Izlaz: Sljedeći token ili klasifikacija
   
✅ Govor → tekst:
   Ulaz: [audio_frame1, audio_frame2...]
   Izlaz: Tekst
   
✅ Stanja sistema (kao gas turbine):
   Ulaz: Sekvenca mjere (451 vremenskih koraka)
   Izlaz: Buduća snaga
```

---

## 🔄 **ZAŠTO RNN ZA GAS TURBINE?**

### Problem: Snaga turbine ovisi o POVIJESTI

```
Scenario A (ANN bi loše obradio):
  Ulaz: Samo V(t) = 5.5V
  Izlaz: P = ???
  
  Problem: Ne znam kontekst! Je li snaga brzo rasla? Padala?
           Gdje je bila prije 1 sekunde?

Scenario B (RNN ide perfektno):
  Ulaz: Sekvenca [V(t-450), V(t-449), ..., V(t-1), V(t)]
  
  RNN vidi:
    - Trend: Raste li napetost?
    - Brzina promjene: Kako brzo se mijenja?
    - Uzorak: Postoji li periodičan uzorak?
    - Inercija: Sistem ima "sjećanje" (masa, kapaciteta, itd)
  
  Izlaz: P(t+1) = mnogo preciznije!
```

### Matematika snage turbine

```
P(t+1) NIJE samo funkcija V(t):
  P(t+1) = f(V(t), V(t-1), V(t-2), ... , V(t-450))
                           ↑
                   ČITAVA SEKVENCA BITNA!
                   
To je razlog zašto:
  - ANN NEĆE raditi dobro
  - RNN ĆE raditi odličnoESTIT
  - LSTM/GRU (RNN varijante) ĆE raditi najbolje
```

---

## 🧠 **LSTM - Poboljšana RNN**

RNN ima problem: **vanishing gradient** - preduge sekvence = težak trening.

**LSTM (Long Short-Term Memory)** rješava to sa **cell state** (dugoročna memorija).

```
Što je različito od običnog RNN-a?

RNN:  h[t] = tanh(...)           (samo kratkoročna memorija)

LSTM: c[t] = f_t * c[t-1] + i_t * c_tilde[t]  (dugoročna memorija!)
      h[t] = o_t * tanh(c[t])
      
Gdje su f_t, i_t, o_t "kapije" koje kontroliraju informacijski tok.
```

**Zašto koristimo LSTM u ovom projektu?**
- Gas turbine imaju **inerciju** (dugoročnu memoriju)
- Trebamo pamtiti stanja iz 451 vremenskog koraka
- LSTM bolje rukuje dugim sekvencama

---

## 📊 **USPOREDBA: NAŠA PROJEKTA**

```
NAŠA SITUACIJA:

Ulaz: Sekvenca napetosti iz 451 vremenskog koraka
      [V(t-450), V(t-449), ..., V(t-1), V(t)]
      
Izlaz: Snaga turbine u sljedećem koraku
      P(t+1)

Zašto RNN (LSTM)?
  ✅ Sekvencijalni ulaz (451 koraka!)
  ✅ Vremenski ovisna predikcija
  ✅ Sistem ima memoriju (inercija turbine)
  ✅ Potrebno učenje dugih ovisnosti (LSTM za to!)
  
Zašto NE ANN?
  ❌ Ne može obraditi 451-dimenzijski vektor direktno
  ❌ Ne razumije vremenski redoslijed
  ❌ Ne pamti prethodna stanja
```

---

## 🎯 **SAŽETAK**

| Svojstvo | ANN | RNN | LSTM |
|----------|-----|-----|------|
| **Arhitektura** | Feed-forward | Petlje | Petlje + gate |
| **Memorija** | ❌ | ✅ | ✅✅ |
| **Za sekvence** | ❌ | ✅ | ✅✅ |
| **Za duge sekvence** | ❌ | ⚠️ | ✅ |
| **Brzina** | ✅ | ⚠️ | ⚠️ |
| **Za gas turbine** | ❌ | ✅ | ✅✅ |

---

## 🚀 **ZAKLJUČAK**

```
ANN:  "Samo mi reci što je sada. Zaboravim prošlost."
RNN:  "Trebam znati što je bilo. Koristim prošlost za sada."
LSTM: "Trebam znati što je bilo DAVNO. Pamtim bitne stvari."

Za gas turbine: LSTM > RNN >> ANN
```

---

**Krerano:** 2026-07-01  
**Vezano na projekt:** Objašnjenje zašto koristimo RNN umjesto ANN

