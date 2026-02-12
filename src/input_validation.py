from dataclasses import dataclass
from models import UserInput
from hardware_profile import HardwareProfil
ValidationResult = <NODE:12>Unsupported Node type: 12
()

def validate_user_input(user_input = None, hw_profile = None):
    '''Pr├╝ft GPU-/Aufl├Âsungs-/FPS-Eingaben auf Plausiblit├ñt ung gibt Hinweise.'''
    messages = []
    corrected = UserInput(gpu_name = user_input.gpu_name.strip(), resolution = user_input.resolution.strip(), target_fps = user_input.target_fps)
    if 'x' not in corrected.resolution:
        messages.append('Die Aufl├Âsung sollte im Format BreitexH├Âhe angegeben werden, z.B. 1920x1080 Du findest die aktuelle Aufl├Âsung in den Windows Anzeigeeinstellungen.')
    if corrected.target_fps <= 0 or corrected.target_fps > 240:
        messages.append('Die Ziel-FPS wirken unplausibel. ├£blich sind z.B. 30, 60, oder 144 FPS Schau in den Monitoreinstellungen nach der Bildwiederholungsrate (Hz).')
    if corrected.gpu_name != 'Unbekannte GPU' and corrected.gpu_name.lower() not in hw_profile.gpu.lower():
        messages.append('Die eingegebene GPU stimmt nciht mit der erkannten GPU ├╝berein. Den exakten Namen findest du im Ger├ñte-Manager oder im Grafikkarten-Treiber.')
    is_valid = len(messages) == 0
    if not is_valid:
        corrected = None
    return ValidationResult(is_valid = is_valid, messages = messages, corrected_input = corrected)