from dataclasses import dataclass
from typing import Literal
from scoring import ScoreResult
FpsPrediction = <NODE:12>Unsupported Node type: 12
()

def predict_fps_from_scoring(score_result = None):
    details = score_result.details
    score = score_result.score
    gpu_vs_min = details.get('gpu_vs_min', 1)
    res_penalty = details.get('res_fps_penalty', 1)
    ram_factor = details.get('ram_factor', 1)
    target_fps = details.get('target_fps', 60)
    base_fps = 120 * min(gpu_vs_min, 2) * res_penalty * ram_factor
    adjusted_fps = base_fps * (score / 100) * (target_fps / 60) ** 0.6
    if adjusted_fps < 55:
        fps_class = '<60'
        confidence = min(0.85, 0.6 + score / 200)
        predicted_range = f'''30-{int(adjusted_fps + 15)} FPS'''
        recommendation = 'Niedrige Einstellungen + Upscaling'
    elif adjusted_fps < 105:
        fps_class = '60-100'
        confidence = 0.88
        predicted_range = f'''{int(adjusted_fps - 20)}-{int(adjusted_fps + 25)} FPS'''
        recommendation = 'Mittlere Einstellungen + Upscaling'
    else:
        fps_class = '>100'
        confidence = min(0.92, 0.75 + gpu_vs_min / 4)
        predicted_range = f'''>{int(adjusted_fps - 15)} FPS'''
        recommendation = 'H├Âhere Einstellungen'
    return (fps_class, confidence, predicted_range, recommendation)

PS C:\Users\Der Don D\Documents\Repos\GamingPlayabilityAssistant> .\pycdc.exe .\game_profile.cpython-312.pyc
# Source Generated with Decompyle++
# File: game_profile.cpython-312.pyc (Python 3.12)


class GameProfil:

    def __init__(self, game_name, min_cpu, min_gpu = None, min_ram = None, rec_cpu = None, rec_gpu = (None, None, None), rec_ram = ('game_name', str, 'min_cpu', str, 'min_gpu', str, 'min_ram', int, 'rec_cpu', str | None, 'rec_gpu', str | None, 'rec_ram', int | None)):
        self.game_name = game_name
        self.min_cpu = min_cpu
        self.min_gpu = min_gpu
        self.min_ram = min_ram
        self.rec_cpu = rec_cpu
        self.rec_gpu = rec_gpu
        self.rec_ram = rec_ram


    def __repr__(self = None):
        return f'''GameProfil(name={self.game_name}, min_cpu={self.min_cpu}, min_gpu={self.min_gpu}, min_ram={self.min_ram}, rep_cpu={self.rec_cpu}, rep_gpu={self.rec_gpu}, rep_ram={self.rec_ram})'''