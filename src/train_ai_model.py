import os
import torch
import torch.nn as nn
import torch.optim as optim
from playability_data import playability_data

# Netzwerk
class PlayabilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Modell, Optimizer, Loss
model = PlayabilityNet()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # etwas kleiner als 0.001
loss_fn = nn.MSELoss()

# Gewichte besser initialisieren
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.zeros_(p)

# Bereich für jedes Feature (du kannst das später anpassen)
GPU_MIN, GPU_MAX = 1.0, 4.0
CPU_MIN, CPU_MAX = 1.0, 2.5
RAM_MIN, RAM_MAX = 8, 64
RES_MIN, RES_MAX = 1920*1080, 3840*2160
FPS_MIN, FPS_MAX = 30, 144

def normalize_gpu_score(score):
    return (score - GPU_MIN) / (GPU_MAX - GPU_MIN)

def normalize_cpu_score(score):
    return (score - CPU_MIN) / (CPU_MAX - CPU_MIN)

def normalize_ram(ram):
    return (ram - RAM_MIN) / (RAM_MAX - RAM_MIN)

def normalize_resolution_pixels(pixels):
    return (pixels - RES_MIN) / (RES_MAX - RES_MIN)

def normalize_target_fps(fps):
    return (fps - FPS_MIN) / (FPS_MAX - FPS_MIN)

# Training
print(f"Starte Training mit {len(playability_data)} Datensätzen...")

for epoch in range(50):
    total_loss = 0.0

    for sample in playability_data:
        # Features normalisieren
        features = torch.tensor([
            normalize_gpu_score(sample["gpu_score"]),
            normalize_cpu_score(sample["cpu_score"]),
            normalize_ram(sample["ram"]),
            normalize_resolution_pixels(sample["resolution_pixels"]),
            normalize_target_fps(sample["target_fps"])
        ], dtype=torch.float32).unsqueeze(0)

        target = torch.tensor([[sample["playability_index"]]], dtype=torch.float32)

        pred = model(features)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(playability_data)
    print(f"Epoch {epoch+1:2d} | Durchschnitts-Loss: {avg_loss:.6f}")

# Ordner erstellen und Modell speichern
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "playability_net.pt")
torch.save(model.state_dict(), model_path)
print(f"\nModell gespeichert unter: {model_path}")
