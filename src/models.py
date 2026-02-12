from dataclasses import dataclass


@dataclass
class UserInput:
    gpu_name: str
    resolution: str
    target_fps: int