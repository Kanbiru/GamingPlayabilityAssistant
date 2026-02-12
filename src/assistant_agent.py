from dataclasses import dataclass
from typing import Dict
from scoring import ScoreResult
from ai_prediction import predict_performance


@dataclass
class AssistantDecision:
    # Entscheidung des Assistenten: Endbewertung, Label, Erklärungen und Empfehlungen
    final_score: int
    final_label: str
    explanations: list[str]
    recommendations: list[str]

    # Optional: KI-Prognose, Label und Vergleich
    ai_index: float | None = None
    ai_label: str | None = None
    ai_agreement: str | None = None


def _ai_index_to_label(x: float) -> str:
    """
    Wandelt den KI-Prognose-Index in ein verständliches Label um.
    """
    if x < 0.3:
        return "KI-Prognose: Kaum flüssig spielbar."
    elif x < 0.55:
        return "KI-Prognose: Nur eingeschränkt spielbar."
    elif x < 0.8:
        return "KI-Prognose: Gut spielbar."
    else:
        return "KI-Prognose: Sehr gut spielbar."


def _ai_agreement(score: int, ai_index: float) -> str:
    """
    Vergleicht regelbasierte Bewertung und KI-Prognose und gibt Übereinstimmung zurück.
    """
    diff = abs((score / 100.0) - ai_index)

    if diff < 0.1:
        return "Hohe Übereinstimmung zwischen KI-Prognose und Systembewertung."
    elif diff < 0.25:
        return "Teilweise Abweichung zwischen KI-Prognose und Systembewertung."
    else:
        return "Abweichende Einschätzung der KI-Prognose."


def run_assistant(score_result: ScoreResult) -> AssistantDecision:
    """
    Hauptfunktion: Bewertet die Spielbarkeit anhand von ScoreResult und KI-Prognose.
    Gibt eine strukturierte Entscheidung mit Erklärungen und Empfehlungen zurück.
    """
    d: Dict[str, float] = score_result.details  # Details aus der Bewertung
    explanations: list[str] = []  # Liste für Erklärungen
    recommendations: list[str] = []  # Liste für Empfehlungen

    # Ziel-FPS aus den Details extrahieren
    target_fps = int(d.get("target_fps", 60))

    # Werte für GPU, CPU, RAM, Auflösung und WinSAT extrahieren
    gpu_vs_min = d.get("gpu_vs_min", 1.0)
    gpu_vs_rec = d.get("gpu_vs_rec", 1.0)
    cpu_vs_min = d.get("cpu_vs_min", 1.0)
    cpu_vs_rec = d.get("cpu_vs_rec", 1.0)
    ram_factor = d.get("ram_factor", 1.0)
    res_penalty = d.get("res_fps_penalty", 1.0)
    winsat_total = d.get("winsat_total")
    winsat_cpu = d.get("winsat_cpu")
    winsat_gpu = d.get("winsat_gpu")
    winsat_memory = d.get("winsat_memory")

    # KI-Prognose berechnen
    ai_res = predict_performance(gpu_vs_min, cpu_vs_min, ram_factor, res_penalty, target_fps)
    ai_index = ai_res.playability_index

    # 1) Erklärung zur Hardwarelage
    # GPU
    if gpu_vs_min < 1.0:
        explanations.append("Deine Grafikkarte liegt unter den Mindestanforderungen des Spiels.")
    elif gpu_vs_min < 1.3:
        explanations.append("Deine Grafikkarte liegt nur leicht über den Mindestanforderungen.")
    elif gpu_vs_rec < 1.0:
        explanations.append("Deine Graffikkarte erfüllt die Mindestanforderungen, verfehlt aber die Empfehlung.")
    elif gpu_vs_rec < 1.3:
        explanations.append("Deine Graffikkarte ist deutlich stärker als die Mindestanforderungen und nahe an den Empfehlungen.")

    # CPU
    if cpu_vs_min < 1.0:
        explanations.append("Deine CPU ist schwächer als die Mindestanforderungen und könnte limitieren.")
    elif cpu_vs_rec < 1.0:
        explanations.append("Deine CPU liegt im Bereich der Mindestanforderungen.")
    else:
        explanations.append("Deine CPU bietet Reserven über den Mindestanforderungen.")

    # RAM
    if ram_factor < 1.0:
        explanations.append("Der verfügbare RAM ist knapp und kann zu Nachladern und Rucklern führen.")
    elif ram_factor > 1.1:
        explanations.append("Du hast mehr RAM als empfohlen, was für dieses Spiel komfortabel ist.")

    # Auflösung/FPS
    if res_penalty < 1.0:
        explanations.append("Hohe Auflösung und/oder hohe Ziel-FPS erhöhen die Last auf GPU und CPU.")

    if target_fps >= 120:
        explanations.append(f"Du strebst sehr hohe {target_fps} FPS an; das erhöht die Anforderungen deutlich und macht Kompromisse bei der Grafikqualität notwendig.")
    elif target_fps <= 45:
        explanations.append(f"Mit einer eher niedrigen Ziel-FPS von {target_fps} kannst du etwas mehr Grafikqualität zulassen, ohne dass die Performance stark leidet.")
    else:
        explanations.append(f"Eine Ziel-FPS von {target_fps} ist ein guter Kompromiss zwischen Bildqualität und flüssigem Gameplay.")

    # WinSAT-Gesamtbewertung
    if winsat_total is not None:
        if winsat_total < 5:
            explanations.append("Der Windows-Leistungsindex bewertet dein System insgesamt als schwach.")
        elif winsat_total < 7:
            explanations.append("Der Windows-Leistungsindex stuft dein System als solide Mittelklasse ein.")
        else:
            explanations.append("Der Windows-Leistungsindex bestätigt eine hohe Gesamtleistung deines Systems.")

    if winsat_gpu is not None and winsat_gpu >= 7:
        explanations.append("Auch laut Windows-Leistungsindex ist deine Grafikkarte überdurchschnittlich leistungsstark.")

    # 2) Konkrete Empfehlungen basierend auf Score
    if score_result.score < 30:
        recommendations.append("Stelle alle Grafikeinstellungen auf niedrig und reduziere Auflösung und Ziel-FPS deutlich.")
        recommendations.append("Wenn möglich, plane ein Upgrade von GPU und RAM.")
        if target_fps > 60:
            recommendations.append(f"Reduziere deine Ziel-FPS von {target_fps} auf etwa 30-45, um überhaupt ein ruhiges Bild zu erreichen.")
    elif score_result.score < 60:
        recommendations.append("Nutze überwiegend niedrige bis mittlere Grafikeinstellungen und deaktiviere aufwändige Effekte wie Raytracing.")
        recommendations.append("Reduziere die Auflösung (z.B. von 4k auf 1440p), falls du Ruckler bemerkst.")
        if target_fps >= 120:
            recommendations.append(f"Für {target_fps} FPS solltest du zunächst die Ziel-FPS auf 60 senken und anschließend schrittweise erhöhen.")
    elif score_result.score < 80:
        recommendations.append("Du kannst mit mittleren bis hohen Einstellungen spielen; teste höhere Settings schrittweise.")
        recommendations.append("Falls du FPS-Einbrüche siehst, reduziere zuerst Schattenqualität und Post-Processing.")
        if target_fps <= 45:
            recommendations.append(f"Wenn dir das Spiel zu träge wirkt, erhähe deine Ziel-FPS von {target_fps} auf 60 und passe die Grafik etwas nach unten an.")
    else:
        recommendations.append("Deine Hardware ist sehr gut geeignet: Hohe bis sehr hohe Einstellungen (z.B. Ultra) und hohe Auflösungen sind realistisch.")
        recommendations.append("Du kannst zusätzliche Effekte (z.B. Raytracing) aktivieren und danach die FPS beobachten.")
        if target_fps >= 120:
            recommendations.append(f"Bei {target_fps} Ziel-FPS kannst du ein hohes Preset nutzen; beobachte aber die Frametime und reduziere Effekte, falls es zu Einbrüchen kommt.")

    # Finaler Agent-Score berechnen (heuristisch und WinSAT)
    final_score = score_result.score
    if winsat_total is not None:
        # 60% heuristischer Score, 40% WinSAT (0-10 -> 0-100)
        final_score = round(0.6 * final_score + 0.4 * winsat_total * 10)

    # Label anhand des finalen Scores bestimmen
    if final_score < 40:
        final_label = "Unter Mindestanforderungen / weniger spielbar"
    elif final_score < 65:
        final_label = "Spielbar mit deutlichen Einschränkungen (niedrige Einstellungen empfohlen)"
    elif final_score < 85:
        final_label = "Gut spielbar (mittlere bis hohe Einstellungen möglich)"
    else:
        final_label = "Sehr gut spielbar (hohe bis ultra Einstellungen möglich)"

    # KI-Prognose interpretieren
    ai_label = None
    ai_agreement = None
    if ai_index is not None:
        ai_label = _ai_index_to_label(ai_index)
        ai_agreement = _ai_agreement(score_result.score, ai_index)
        explanations.append(f"Die KI-basierte Performance-Prognose stuft das Spiel als {ai_label} ein.")

        if ai_agreement == "Abweichende Einschätzung der KI-Prognose.":
            explanations.append(
                "Die KI-Prognose weicht deutlich von der regelbasierten Bewertung ab."
                "In solchen Fällen sind individuelle Spiel- und Treiberfaktoren besonders relevant."
            )

    # Entscheidung zurückgeben
    return AssistantDecision(
        final_score=final_score,
        final_label=final_label,
        explanations=explanations,
        recommendations=recommendations,
        ai_index=ai_index,
        ai_label=ai_label,
        ai_agreement=ai_agreement,
    )
