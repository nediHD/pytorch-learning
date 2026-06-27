"""
Vollständiger PyTorch Trainings-Loop für das Gas Turbine Prediction Modell.

Dies ist der zentrale Lernort. Jeder Abschnitt erklärt ein PyTorch-Konzept.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import GasTurbineDataset
from model import WindowMLP


# ============================================================
# 0. HYPERPARAMETER — Leicht veränderbar zum Experimentieren
# ============================================================
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8  # 80% Training, 20% Validation


def main():
    # Stelle sicher, dass der saved_models Ordner existiert
    os.makedirs('saved_models', exist_ok=True)

    # ============================================================
    # 1. DATEN LADEN
    # ============================================================
    print("=== 1. Loading Data ===\n")

    dataset = GasTurbineDataset('data/train', window_size=WINDOW_SIZE)

    # ============================================================
    # 2. TRAIN/VALIDATION SPLIT
    # ============================================================
    print("=== 2. Train/Validation Split ===\n")

    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # Das ist wichtig: Das Validierungsset sieht der Modell während des Trainings NICHT.
    # Damit können wir überprüfen, ob das Modell wirklich gelernt hat oder nur die
    # Trainsdaten auswendig gelernt hat (Overfitting).

    # ============================================================
    # 3. DATALOADERS ERSTELLEN
    # ============================================================
    print("=== 3. Creating DataLoaders ===\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # Shuffle: zufällige Reihenfolge beim Training, besser zum Lernen
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False  # Keine Shuffle für Validation, um konsistent zu bleiben
    )

    print(f"Batches per epoch: {len(train_loader)}\n")

    # Der DataLoader ist ein Iterator, der Daten in Batches gibt.
    # for X_batch, y_batch in train_loader: würde Batches von Größe BATCH_SIZE geben.

    # ============================================================
    # 4. MODELL, LOSS-FUNKTION, OPTIMIZER
    # ============================================================
    print("=== 4. Model, Loss, Optimizer ===\n")

    model = WindowMLP(window_size=WINDOW_SIZE)
    print(f"Model:\n{model}\n")

    # Loss: nn.MSELoss = Mean Squared Error
    # Für Regression (kontinuierliche Vorhersagen) ist MSE der Standard.
    # loss = (1/n) * Σ(predicted - actual)^2
    criterion = nn.MSELoss()

    # Optimizer: optim.Adam
    # Adam = Adaptive Moment Estimation, eine moderne Variante von Stochastic Gradient Descent.
    # Er verwaltet automatisch die Lernrate pro Parameter.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ============================================================
    # 5. TRAININGS-LOOP
    # ============================================================
    print("=== 5. Training Loop ===\n")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()  # Setzt das Modell in Training-Modus (wichtig für Dropout, BatchNorm, etc.)

        train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # optimizer.zero_grad(): Alte Gradienten löschen
            # (sonst würden sie sich akkumulieren)
            optimizer.zero_grad()

            # Forward Pass: X durch das Modell
            predictions = model(X_batch)

            # Loss berechnen
            loss = criterion(predictions.squeeze(), y_batch)

            # Backward Pass: Gradienten berechnen
            # loss.backward() nutzt Autograd: dL/dW für alle Parameter
            loss.backward()

            # Optimizer Step: Gewichte updaten
            # optimizer.step() macht einen Gradient Descent Schritt: W := W - lr * dL/dW
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation Phase ---
        model.eval()  # Setzt das Modell in Evaluation-Modus

        val_loss = 0.0
        # torch.no_grad(): Keine Gradienten berechnen während Validation
        # Das spart Memory und Rechenzeit, da wir die Gewichte nicht updaten.
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print()

    # ============================================================
    # 6. MODELL SPEICHERN
    # ============================================================
    print("=== 6. Saving Model ===\n")

    checkpoint = {
        'model_state_dict': model.state_dict(),  # Alle gelernten Gewichte
        'window_size': WINDOW_SIZE,
        'input_min': dataset.input_min,
        'input_max': dataset.input_max,
        'target_min': dataset.target_min,
        'target_max': dataset.target_max,
    }

    model_path = 'saved_models/model.pth'
    torch.save(checkpoint, model_path)

    print(f"Model saved to {model_path}\n")

    # state_dict() ist ein dict mit allen trainierten Parametern (Gewichte + Bias).
    # Wir speichern auch die Normalisierungswerte, damit predict.py sie verwenden kann.
    # weights_only=True wird zum Standard in PyTorch — es verhindert Code-Injection Angriffe.


if __name__ == '__main__':
    main()

# === NÄCHSTE SCHRITTE für Lernende ===
# 1. Erhöhe EPOCHS auf 100 — ändert sich die Val Loss noch oder plateaued sie?
#    (Wenn sie plateaued: Saturation. Wenn sie steigt: Overfitting.)
# 2. Verändere LEARNING_RATE: Zu hoch (z.B. 0.1) → Instabilität. Zu niedrig → Langsames Lernen.
# 3. Versuche einen Learning Rate Scheduler: optim.lr_scheduler.StepLR
# 4. Plotte Train Loss und Val Loss über die Epochs (verwende matplotlib).
