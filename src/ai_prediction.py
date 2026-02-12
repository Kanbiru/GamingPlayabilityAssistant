import torch
import torch.nn as nn
from dataclasses import dataclass

# Datenklasse für das Ergebnis der KI-Prognose
@dataclass
class AiResult:
    playability_index: float  # Bewertung der Spielbarkeit (0 bis 1)


# Mini-Modell für die KI-Prognose der Spielbarkeit
class PlayabilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Neuronales Netz mit mehreren Schichten: Eingabe (5), versteckte Schichten, Ausgabe (1)
        self.net = nn.Sequential(
            nn.Linear(5, 16),   # Erste Schicht: 5 Eingaben, 16 Neuronen
            nn.ReLU(),          # Aktivierungsfunktion
            nn.Linear(16, 8),   # Zweite Schicht: 16 Eingaben, 8 Neuronen
            nn.ReLU(),          # Aktivierungsfunktion
            nn.Linear(8, 1),    # Ausgabeschicht: 8 Eingaben, 1 Neuron
            nn.Sigmoid()        # Ausgabe zwischen 0 und 1
        )

    def forward(self, x):
        # Vorwärtsdurchlauf: Eingabe durch das Netz leiten
        return self.net(x)


# Vorinitialisiertes Modell instanziieren
model = PlayabilityNet()


# Gewichte mit festem Seed initialisieren (reproduzierbar)
torch.manual_seed(42)
for p in model.parameters():
    nn.init.uniform_(p, -0.5, 0.5)  # Gewichte gleichmäßig zwischen -0.5 und 0.5 setzen
model.eval()  # Modell in Evaluierungsmodus setzen (keine Trainingseffekte)


def predict_performance(gpu_vs_min, cpu_vs_min, ram_factor, res_penalty, target_fps) -> AiResult:
    """
    Berechnet eine KI-basierte Vorhersage der Spielbarkeit basierend auf gegebenen Features.
    Keine Dummy-Daten, keine Datensammlung nötig.
    """
    # Feature-Vektor aus den übergebenen Werten erstellen
    x = torch.tensor([gpu_vs_min, cpu_vs_min, ram_factor, res_penalty, target_fps], dtype=torch.float32)
    x = x.unsqueeze(0)  # Batch-Dimension hinzufügen (erforderlich für das Modell)
    with torch.no_grad():  # Keine Gradienten berechnen (nur Vorhersage)
        y = model(x)
    # Ergebnis als AiResult zurückgeben
    return AiResult(playability_index=float(y.item()))
