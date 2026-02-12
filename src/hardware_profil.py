from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfil:
    cpu: str
    gpu: str
    ram: int
    aufloesung: str
    ziel_fps: int
    winsat_cpu: Optional[float] = None
    winsat_gpu: Optional[float] = None
    winsat_memory: Optional[float] = None
    winsat_total: Optional[float] = None


    # Terminalausgaben
    def __repr__(self) -> str:
        return(
            f"HardwareProfil(cpu={self.cpu}, gpu={self.gpu}, "
            f"ram={self.ram}, aufloesung={self.aufloesung}, "
            f"ziel_fps={self.ziel_fps}"
            f"winsat_cpu={self.winsat_cpu}, winsat_gpu={self.winsat_gpu}, "
            f"winsat_memory={self.winsat_memory}, winsat_total={self.winsat_total})"
        )