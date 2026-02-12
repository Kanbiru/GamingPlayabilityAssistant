# Dictionaries zur Bewertung verschiedener GPU- und CPU-Modelle
GPU_SCORES = {
    "gtx 1060": 5,
    "rtx 2060": 6,
    "rtx 3060": 7,
    "rtx 4070": 8,
    "rtx 5070": 9,
}

CPU_SCORES = {
    "i5-6700": 5,
    "i7-6700": 6,
    "i7-12700": 7,
    "ryzen 5 1600": 5,
    "ryzen 7 7800x3d": 9,
}


def estimate_gpu_score(name: str | None) -> int:
    """
    Schätzt den GPU-Score anhand des Namens.
    Gibt eine Zahl zwischen 1 und 9 zurück, basierend auf der Modellbezeichnung.
    Falls kein Name vorhanden oder bekannt, wird ein Fallback-Wert 4 genutzt.
    """
    if name is None:
        return 4  # Fallback, falls kein Name vorliegt
    n = name.lower()
    for key, score in GPU_SCORES.items():
        if key in n:
            return score  # Rückgabe des Scores, wenn Name passt
    return 4  # Fallback-Wert, wenn kein Match gefunden wurde

def estimate_cpu_score(name: str | None) -> int:
    """
    Schätzt den CPU-Score anhand des Namens.
    Gibt eine Zahl zwischen 1 und 9 zurück, basierend auf der Modellbezeichnung.
    Falls kein Name vorhanden oder bekannt, wird ein Fallback-Wert 4 genutzt.
    """
    if name is None:
        return 4  # Fallback, falls kein Name vorliegt
    n = name.lower()
    for key, score in CPU_SCORES.items():
        if key in n:
            return score  # Rückgabe des Scores bei Übereinstimmung
    return 4  # Fallback-Wert, wenn kein Match gefunden wurde
