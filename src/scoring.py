from dataclasses import dataclass
from typing import Dict


from game_profile import GameProfil
from hardware_profile import HardwareProfil
from models import UserInput
from score_utils import estimate_gpu_score, estimate_cpu_score
from ai_prediction import predict_performance



@dataclass
class ScoreResult:
    score: int
    level: str
    details: Dict[str, float]



def _ram_factor(hw_ram: int, min_ram: int, rec_ram: int | None = None) -> float:
    """Berechnet einen RAM-Faktor zwischen 0 und > 1."""
    if hw_ram <= 0 or min_ram is None or min_ram <=0:
        return 1.0 # Fallback


    if hw_ram < min_ram:
        # Unter Mindestanforderung -> stark abwerten
        return hw_ram / min_ram


    # Über Mindestanforderung
    if rec_ram is not None and rec_ram > min_ram:
        if hw_ram >= rec_ram:
            return 1.2
        ratio = (hw_ram - min_ram) / (rec_ram - min_ram)
        return 1.0 + 0.2 * ratio
    else:
        # Es gibt keine Empfehlung -> minimaler Bonus
        return 1.05



def _fps_resolution_penalty(user: UserInput, gpu_vs_min: float) -> float:
    """Grobe Straf-/Bonusfunktion für hohe Auflösung und FPS."""
    # Basis: 1.0 = neutral
    penalty = 1.0


    # Auflösung
    res = user.resolution.lower()


    if "3840x2160" in res or "4k" in res:
        penalty -= 0.15
        if gpu_vs_min >= 1.5:
            penalty += 0.10
    elif "2560x1440" in res or "1440p" in res or "2k" in res:
        penalty -= 0.05
    # 1080p / FullHD bleibt neutral


    #FPS prüfen
    fps = user.target_fps
    if fps > 60:
        penalty -= 0.1
    elif fps < 30:
        # sehr niedrige Ziel-FPS -> etwas Bonus, weil weniger Leistung nötig
        penalty += 0.05
   
    # Untergrenze, damit nie negativ wird
    return max(0.4, penalty)



def resolution_to_pixels(resolution: str) -> int:
    width, height = map(int, resolution.split("x"))
    return width * height
   


def calculate_playability_score(hardware_profile: HardwareProfil, game_profile: GameProfil, user_input: UserInput,) -> ScoreResult:
    """Heuristisches Scoring auf Basis von RAM, einfacher CPU/GPU-Heuristik und einem Strafterm für hohe Auflösung/FPS"""
    details: Dict[str, float] = {}


    if hardware_profile.winsat_total is not None:
        details["winsat_total"] = hardware_profile.winsat_total
        if hardware_profile.winsat_cpu is not None:
            details["winsat_cpu"] = hardware_profile.winsat_cpu
        if hardware_profile.winsat_gpu is not None:
            details["winsat_gpu"] = hardware_profile.winsat_gpu
        if hardware_profile.winsat_memory is not None:
            details["winsat_memory"] = hardware_profile.winsat_memory


    # 1. RAM-Faktorund
    ram_factor = _ram_factor(
        hw_ram=hardware_profile.ram,
        min_ram=game_profile.min_ram,
        rec_ram=game_profile.rec_ram,
    )
    details["ram_factor"] = ram_factor


    # 2. CPU/GPU-Matching (sehr grob: String-Vergleich)
    # GPU
    hw_gpu_score = estimate_gpu_score(hardware_profile.gpu)
    min_gpu_score = estimate_gpu_score(game_profile.min_gpu)
    rec_gpu_score = estimate_gpu_score(game_profile.rec_gpu or game_profile.min_gpu)


    gpu_vs_min = hw_gpu_score / min_gpu_score if min_gpu_score else 1.0
    gpu_vs_rec = hw_gpu_score / rec_gpu_score if rec_gpu_score else 1.0
    details["gpu_vs_min"] = gpu_vs_min
    details["gpu_vs_rec"] = gpu_vs_rec


    # CPU
    hw_cpu_score = estimate_cpu_score(hardware_profile.cpu)
    min_cpu_score = estimate_cpu_score(game_profile.min_cpu)
    rec_cpu_score = estimate_cpu_score(game_profile.rec_cpu or game_profile.min_cpu)


    cpu_vs_min = hw_cpu_score / min_cpu_score if min_cpu_score else 1.0
    cpu_vs_rec = hw_cpu_score / rec_cpu_score if rec_cpu_score else 1.0


    if cpu_vs_min < 1.0 and gpu_vs_min >= 1.5 and ram_factor >= 1.0:
        cpu_vs_min = 1.0


    details["cpu_vs_min"] = cpu_vs_min
    details["cpu_vs_rec"] = cpu_vs_rec


    # 3. Auflösung/FPS-Strafterm
    res_fps_penalty = _fps_resolution_penalty(user_input, gpu_vs_min)
    details["res_fps_penalty"] = res_fps_penalty


    details["target_fps"] = float(user_input.target_fps)


    # 4. Basis-Score berechnen
    # Gewichte
    base = 0.0
    base += 65 * min(gpu_vs_min, 2.0) / 2.0
    base += 15 * min(cpu_vs_min, 2.0) / 2.0
    base += 15 * ram_factor


    # Bonus
    if gpu_vs_min >= 1.0 and cpu_vs_min >= 1.0 and ram_factor >= 1.0:
        base += 5


    # Strafterm für hohe Anforderung
    score = base * res_fps_penalty


    # Score auf 0-100 begrenzen und runden
    score = max(0, min(100, round(score)))
    details["base_before_penalty"] = base


    # 5. Level bestimmen
    if score < 40:
        level = "Unter Mindestanforderungen / weniger spielbar"
    elif score < 65:
        level = "Spielbar mit deutlichen Einschränkungen (niedrige Einstellungen empfohlen)"
    elif score < 85:
        level= "Gut spielbar (mittlere Einstellungen möglich)"
    else:
        level = "Sehr gut spielbar (hohe bis ultra Einstellungen möglich)"


    # --- KI Prediction (ECHT, unabhängig vom Score) ---
    ai = predict_performance(
        hw_gpu_score,
        hw_cpu_score,
        hardware_profile.ram,
        user_input.resolution,
        user_input.target_fps
    )


    details["ai_playability_index"] = ai.playability_index


    return ScoreResult(score=score, level=level, details=details)