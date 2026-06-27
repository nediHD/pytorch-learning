# PyTorch Learning Repository

Eine Sammlung von PyTorch-Projekten für Anfänger. Jedes Projekt stellt ein reales Lernproblem dar und zeigt die Grundkonzepte von PyTorch in Aktion.

## Projekte

### 1. Micro Gas Turbine Electrical Energy Prediction
**Ordner:** `micro-gas-turbine/`

Zeitreihen-Regression: Vorhersage der elektrischen Leistung eines Mikrogasturbine basierend auf der Eingangsspannung.

**Was du lernst:**
- PyTorch `Dataset` und `DataLoader`
- Ein einfaches neuronales Netz (MLP) mit `nn.Module`
- Der komplette Trainings-Loop: `forward`, `backward`, `step`
- Speichern und Laden von trainierten Modellen
- Normalisierung und Data Leakage vermeiden

**Schnellstart:**
```bash
cd micro-gas-turbine
pip install -r requirements.txt
python train.py
python predict.py
```

---

**Erstellt als Einstiegsprojekt für PyTorch-Anfänger.**
