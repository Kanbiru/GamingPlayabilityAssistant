from dataclasses import dataclass
from typing import List
from scoring import ScoreResult 
from hardware_profile import HardwareProfil 
from game_profile import GameProfil


@dataclass
class GraphicsAdvice:
    # Empfehlungen für Grafikeinstellungen
    preset: str  # z.B. "Low", "Medium", "High", "Ultra"
    raytracing: str  # off/on
    upscaling: str  # off, DLSS/FSR Quality
    vsync: str  # off/on oder adaptive
    resolution_hint: str  # Text-Hinweis zur Auflösung
    extra_tips: List[str]  # Stichpunkte zur Feinanpassung
    explanations: List[str]  # Erklärtext zu Fachbegriffen


def derive_graphics_advice(score_result: ScoreResult, hardware: HardwareProfil, game: GameProfil) -> GraphicsAdvice:
    """
    Leitet konkrete Grafikempfehlungen aus ScoreResult, Hardwareprofil und Spielprofil ab.
    """
    d = score_result.details
    # Werte aus den Details extrahieren
    gpu_vs_min = d.get("gpu_vs_min", 1.0)
    cpu_vs_min = d.get("cpu_vs_min", 1.0)
    ram_factor = d.get("ram_factor", 1.0)
    res_penalty = d.get("res_fps_penalty", 1.0)
    target_fps = getattr(hardware, "ziel_fps", 60)  # Ziel-FPS aus Hardwareprofil

    # Standardwerte für Grafikeinstellungen
    preset = "Medium"
    raytracing = "Off"
    upscaling = "Off"
    vsync = "On"
    resolution_hint = ""
    extra_tips: List[str] = []
    explanations: List[str] = []

    # Regeln basierend auf Score und Hardware
    if score_result.score >= 85 and gpu_vs_min >= 1.5 and cpu_vs_min >= 1.2 and ram_factor >= 1.0:
        # Sehr gute Hardware: Ultra-Einstellungen möglich
        preset = "High/Ultra"
        raytracing = "On (Medium)"
        upscaling = "Optional (Quality)"
        resolution_hint = "4k möglich; bei FPS-Einbrüchen auf 1440p reduzieren."
    elif score_result.score >= 70:
        # Gute Hardware: Mittlere bis hohe Einstellungen
        preset = "Medium/High"
        raytracing = "Off oder nur selektiv"
        upscaling = "On (Quality)"
        resolution_hint = "FullHD gut geeignet; 4k nur mit Einschränkungen."
    elif score_result.score >= 50:
        # Mittlere Hardware: Niedrige bis mittlere Einstellungen
        preset = "Low/Medium"
        raytracing = "Off"
        upscaling = "On (Balanced/Performance)"
        resolution_hint = "1080p empfohlen; Ziel-FPS ggf. auf 30-60 begrenzen."
    else:
        # Schwache Hardware: Niedrige Einstellungen
        preset = "Low"
        raytracing = "Off"
        upscaling = "On (Performance)"
        resolution_hint = "1080p oder darunter; Ziel-FPS eher bei 30 ansetzen."

    # Anpassungen bei sehr hohen oder niedrigen Ziel-FPS
    if target_fps >= 120:
        # Performance-Fokus: niedrigere Presets und aggressiveres Upscaling
        if preset == "High/Ultra":
            preset = "Medium/High"
        elif preset == "Medium/High":
            preset = "Medium"
        upscaling = "On (Balanced/Performance)"
        extra_tips.append(f"Da du {target_fps} FPS als Ziel gewählt hast, sind etwas niedrigere Presets sinnvoll, um Framedrops zu vermeiden.")
        if "3840x2160" in (hardware.aufloesung or ""):
            extra_tips.append("Für sehr hohe FPS bei 4k kann es helfen, auf 1440p oder 1080p zu wechseln.")
    elif target_fps <= 45:
        # Qualitäts-Fokus: Details können höher sein
        extra_tips.append(f"Mit einer eher niedrigen Ziel-FPS von {target_fps} kannst du Grafikdetails etwas höher ansetzen, ohne dass die Performance zu stark leidet.")

    # Allgemeine Tipps zur Feinanpassung
    extra_tips.append("Reduziere zuerst Schattenqualität und volumetrische Effekte, falls FPS einbrechen.")
    extra_tips.append("Deaktiviere Motion Blur und Tiefenunschärfe, wenn das Bild unruhig oder unscharf wirkt.")
    if res_penalty < 1.0:
        extra_tips.append("Hohe Auflösung und hohe Ziel-FPS belasten deine GPU stark; senke zuerst die Auflösung oder FPS.")

    # Erklärungen zu Grafiktechniken
    explanations.append("Raytracing berechnet Licht- und Schatteneffekte realistischer, kostet aber viel GPU-Leistung.")
    explanations.append("V-Sync synchronisiert die Bildrate mit der Monitorfrequenz, verhindert Tearing, kann aber Eingabeverzögerung erhöhen.")
    explanations.append("Upscaling (z.B. DLSS/FSR/XeSS) rendert intern in niedriger Auflösung und skaliert hoch, um mehr FPS zu erreichen.")
    explanations.append("Grafik-Presets (Niedrig/Mittel/Hoch/Ultra) fassen viele Detailsoptionen zusammen; höhere Presets erhöhen Bildqualität, aber senken FPS.")

    # Empfehlungen zurückgeben
    return GraphicsAdvice(
        preset=preset,
        raytracing=raytracing,
        upscaling=upscaling,
        vsync=vsync,
        resolution_hint=resolution_hint,
        extra_tips=extra_tips,
        explanations=explanations,
    )
