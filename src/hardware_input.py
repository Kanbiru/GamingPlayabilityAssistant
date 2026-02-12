import psutil
import platform
import subprocess


from hardware_profile import HardwareProfil
from models import UserInput
from winsat_reader import read_winsat


def detect_cpu_name() -> str:
    """
    Ermittelt den Namen der CPU über das Betriebssystem.
    """
    name = platform.processor() or platform.machine()
    return name or "Unbekannte CPU"


def detect_ram_gb() -> int:
    """
    Liest die installierte RAM-Menge in GB aus.
    """
    vm = psutil.virtual_memory()
    return int(vm.total / (1024 ** 3))


def detect_gpu_name_raw() -> str:
    """
    Versucht, die GPU unter Windows über PowerShell und WMIC zu ermitteln.
    """
    try:
        cmd = [
            "powershell",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        ]
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        return [l.strip() for l in output.splitlines() if l.strip()]
    except Exception:
        return []


def detect_gpu_name() -> str:
    """
    Filtert die GPU-Liste nach NVIDIA/AMD und gibt den Namen zurück.
    Priorisiert dedizierte GPUs.
    """
    names = detect_gpu_name_raw()
    if not names:
        return "Unbekannte GPU"

    for name in names:
        lower = name.lower()
        if "nvidia" in lower or ("radeon" in lower and "tm" not in lower):
            if "geforce" in lower:
                return "NVIDIA " + name.split("GeForce", 1)[1].strip()
            return name
    return names[-1]


# === CLI-Variante ===


def ask_user_gpu_name() -> str:
    """
    Fragt den Nutzer nach dem GPU-Namen (CLI-Eingabe).
    """
    gpu = input("Bitte gib deine Grafikkarte an (z.B. 'NVIDIA RTX 3060'): ").strip()
    return gpu or "Unbekannte GPU"


def ask_user_resolution() -> str:
    """
    Fragt den Nutzer nach der gewünschten Auflösung (CLI-Eingabe).
    """
    print("Gewünschte Auflösung (z.B. FullHD (1920x1080), 2k (2560x1440), 4k (3840x2160)")
    res = input("Auflösung: ").strip()
    return res or "1920x1080"


def ask_user_target_fps() -> int:
    """
    Fragt den Nutzer nach der Ziel-FPS (CLI-Eingabe).
    """
    print("Gewünschte Ziel-FPS (z.B. 30, 60, 120):")
    try:
        fps = int(input("Ziel-FPS: ").strip())
    except ValueError:
        fps = 60
    return fps


def read_hardware_and_user_input() -> tuple[HardwareProfil, UserInput]:
    """
    Erfasst Systemhardware und Roh-User-Eingaben, prüft aber noch nichts.
    """
    cpu_name = detect_cpu_name()
    ram_gb = detect_ram_gb()
    winsat = read_winsat()

    print(f"Erkannte CPU: {cpu_name}")
    print(f"Erkannte RAM: {ram_gb} GB")
    if winsat.total_score is not None:
        print(f"WinSAT-Gesamtscore: {winsat.total_score}")

    gpu_name = ask_user_gpu_name()
    resolution = ask_user_resolution()
    target_fps = ask_user_target_fps()

    hardware_profile = HardwareProfil(
        cpu=cpu_name,
        gpu=gpu_name,
        ram=ram_gb,
        aufloesung=resolution,
        ziel_fps=target_fps,
        winsat_cpu=winsat.cpu_score,
        winsat_gpu=winsat.gpu_score,
        winsat_memory=winsat.memory_score,
        winsat_total=winsat.total_score,
    )

    user_input = UserInput(
        gpu_name=gpu_name,
        resolution=resolution,
        target_fps=target_fps,
    )

    return hardware_profile, user_input


# === UI-Variante ===


def read_hardware_without_cli() -> tuple[HardwareProfil, UserInput]:
    """
    Variante für die GUI: Ermittelt CPU, RAM und WinSAT automatisch.
    GPU, Auflösung und FPS bleiben leer (werden in der GUI gesetzt).
    """
    cpu_name = detect_cpu_name()
    ram_gb = detect_ram_gb()
    gpu_name = detect_gpu_name()
    winsat = read_winsat()

    hardware_profile = HardwareProfil(
        cpu=cpu_name,
        gpu=gpu_name,
        ram=ram_gb,
        aufloesung="",  # Leere Werte als Platzhalter für GUI-Eingabe
        ziel_fps=60,    # Default-Wert
        winsat_cpu=winsat.cpu_score,
        winsat_gpu=winsat.gpu_score,
        winsat_memory=winsat.memory_score,
        winsat_total=winsat.total_score,
    )

    user_input = UserInput(
        gpu_name=gpu_name,
        resolution="",
        target_fps=60,  # Default-Wert
    )

    return hardware_profile, user_input


if __name__ == "__main__":
    hw, ui = read_hardware_and_user_input()
    print("\nErfasstes Hardwareprofil:")
    print(hw)
    print("\nRoh-User-Eingaben:")
    print(ui)
