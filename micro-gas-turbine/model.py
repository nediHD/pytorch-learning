"""
Ein einfaches PyTorch Neural Network (MLP) für Zeitreihen-Regression.

Konzept:
- nn.Module ist die Basisklasse für alle neuronalen Netze in PyTorch
- __init__: Definiere die Schichten (Layer)
- forward(): Definiere wie Daten durch die Schichten fließen
- Das Modell wird wie eine Funktion aufgerufen: y = model(X)
"""

import torch
import torch.nn as nn


class WindowMLP(nn.Module):
    """
    Multi-Layer Perceptron für das Sliding Window Regression Problem.

    Architektur:
    Input (window_size*2) → Hidden(64) → Hidden(32) → Output(1)
    """

    def __init__(self, window_size=30):
        """
        Args:
            window_size (int): Größe des Sliding Windows
                Bestimmt die Input-Größe: window_size * 2
                (weil wir [time_delta, input_voltage] haben)
        """
        super().__init__()

        input_size = window_size * 2

        # Definiere die Schichten
        # nn.Linear(in_features, out_features) ist eine lineare Transformation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # ReLU: Aktivierungsfunktion (Rectified Linear Unit)
        # Macht das Netz nicht-linear: ReLU(x) = max(0, x)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Definiert wie Daten durch das Netz fließen.

        WICHTIG: forward() wird NICHT direkt aufgerufen.
        Stattdessen schreiben wir: y = model(x)
        PyTorch ruft automatisch forward() auf.

        Args:
            x: Input Tensor, Shape (batch_size, window_size*2)

        Returns:
            output: Predictions, Shape (batch_size, 1)
        """
        # Erste Schicht + Aktivierung
        x = self.fc1(x)
        x = self.relu(x)

        # Zweite Schicht + Aktivierung
        x = self.fc2(x)
        x = self.relu(x)

        # Output Schicht (keine Aktivierung bei Regression)
        x = self.fc3(x)

        return x


# === NÄCHSTE SCHRITTE für Lernende ===
# 1. Versuche eine vierte Schicht hinzuzufügen (z.B. 32 → 16 → 1)
# 2. Füge Dropout hinzu um Overfitting zu reduzieren: self.dropout = nn.Dropout(0.2)
# 3. Ersetze die ganze Klasse durch ein LSTM: nn.LSTM(input_size, hidden_size, num_layers)
