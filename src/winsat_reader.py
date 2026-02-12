import subprocess
import json
from dataclasses import dataclass
from typing import Optional
WinSatResult = <NODE:12>Unsupported Node type: 12
()

def read_winsat():
Unsupported opcode: POP_JUMP_IF_NONE (239)
    '''
    F├╝hrt unter Windows den WinSat-Befehl aus und parst die wichtigsten Scores.
    Gibt bei Problemen None-Werte zur├╝ck, damit der Rest des Programms weiterl├ñuft.
    '''
    ps_command = [
        'powershell',
        '-Command',
        'Get-CimInstance Win32_WinSAT | ConvertTo-Json']
    completed = subprocess.run(ps_command, capture_output = True, text = True, check = True)
    output = completed.stdout.strip()
    if not output:
        return WinSatResult(None, None, None, None)
    data = None.loads(output)
    if isinstance(data, list):
        data = data[0]
# WARNING: Decompyle incomplete