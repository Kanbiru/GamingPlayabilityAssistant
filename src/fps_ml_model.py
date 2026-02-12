# fps_ml_model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np

from hardware_profile import HardwareProfil
from models import UserInput

# Pfad zur gespeicherten Modell-Datei (vom Trainings-Skript erzeugt)
_MODEL_PATH = "models/fps_classifier.pkl"


@dataclass
class FpsPrediction:
    fps_class: Literal["<60", "60-100", ">100"]
    proba: float           # Wahrscheinlichkeit für diese Klasse (0-1)
    raw_probas: dict[str, float]  # optionale Detailinfos


_model = None  # wird lazy geladen


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def _encode_resolution(resolution: str) -> int:
    res = (resolution or "").lower()
    if "3840x2160" in res or "4k" in res:
        return 2   # 4k
    if "2560x1440" in res or "1440p" in res or "2k" in res:
        return 1   # 1440p
    return 0       # 1080p / default


def _build_feature_vector(hw: HardwareProfil, ui: UserInput) -> np.ndarray:
    """
    Erzeugt den Feature-Vektor für das ML-Modell aus den vorhandenen Daten.
    Muss konsistent zu dem sein, was im Trainingsskript verwendet wurde.
    Beispiel-Features:
      - gpu_score (wie in scoring.py)
      - cpu_score
      - ram_gb
      - resolution_code (0=1080p, 1=1440p, 2=4k)
      - target_fps
    """
    from scoring import estimate_gpu_score, estimate_cpu_score

    gpu_score = estimate_gpu_score(hw.gpu)
    cpu_score = estimate_cpu_score(hw.cpu)
    ram_gb = hw.ram
    res_code = _encode_resolution(ui.resolution or hw.aufloesung or "1920x1080")
    target_fps = ui.target_fps

    # Reihenfolge muss mit dem Training übereinstimmen
    return np.array([[gpu_score, cpu_score, ram_gb, res_code, target_fps]], dtype=float)


def predict_fps_class(hw: HardwareProfil, ui: UserInput) -> FpsPrediction:
    """
    Gibt eine grobe FPS-Kategorie zurück, basierend auf einem trainierten ML-Modell.
    """
    model = _load_model()
    X = _build_feature_vector(hw, ui)

    proba = model.predict_proba(X)[0]    # z.B. [0.2, 0.6, 0.2]
    y = model.predict(X)[0]              # z.B. 1

    class_map = {0: "<60", 1: "60-100", 2: ">100"}
    fps_class = class_map.get(int(y), "60-100")

    raw_probas = {
        "<60": float(proba[0]),
        "60-100": float(proba[1]),
        ">100": float(proba[2]),
    }

    return FpsPrediction(
        fps_class=fps_class,
        proba=raw_probas[fps_class],
        raw_probas=raw_probas,
    )
