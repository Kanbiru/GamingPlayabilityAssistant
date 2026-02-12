import re
import sys
from hardware_input import read_hardware_and_user_input
from input_validation import validate_user_input
from game_profile import GameProfil
from scoring import calculate_playability_score
from assistant_agent import run_assistant
from api import search_steam_games, fetch_steam_requirements
from graphics_advisor import derive_graphics_advice

def parse_requirements_block(block = None):
    cpu = None
    gpu = None
    ram_gb = None
    if not block:
        return {
            'cpu': None,
            'gpu': None,
            'ram_gb': None }
    for line in None.splitlines():
        line = line.strip().lower()
        if line.lower().startswith('processor:') or line.lower().startswith('cpu:'):
            cpu = line.split(':', 1)[1].strip()
            continue
        if line.lower().startswith('graphics:') or line.lower().startswith('gpu:'):
            gpu = line.split(':', 1)[1].strip()
            continue
        if not line.lower().startswith('memory:') and line.lower().startswith('ram:'):
            continue
        match = re.search('(\\d+)\\s*gb', line.lower())
        if not match:
            continue
        ram_gb = int(match.group(1))
    return {
        'cpu': cpu,
        'gpu': gpu,
        'ram_gb': ram_gb }


def load_game_profile_from_api():
Unsupported opcode: LOAD_FAST_CHECK (237)
    game_name = input('Spielname eingeben: ')
    results = search_steam_games(game_name)
    if not results:
        print('Keine Spiele gefunden.')
        sys.exit(1)
    print('\nGefundene Spiele:')
    for i, g in enumerate(results, start = 1):
        print(f'''{i}. {g['name']} (AppID: {g['appid']})''')
    choice = int(input('\nBitte Nummer aus Liste ausw├ñhlen: '))
    selected = results[choice - 1]
# WARNING: Decompyle incomplete


def main():
    print('===Intelligenter Gaming-Assitent (Prototyp) ===\n')
    (hardware_profile, user_input) = read_hardware_and_user_input()
    print('\n--- Erfasste Hardware ---')
    print(hardware_profile)
    print('\n--- Roh-User-Eingaben ---')
    print(user_input)
    validation_result = validate_user_input(user_input, hardware_profile)
    if not validation_result.is_valid:
        print('\nDer Assitent hat Unstimmigkeiten in deinen Eingaben erkannt:')
        for msg in validation_result.messages:
            print(f'''- {msg}''')
        print('\nBitte korrigiere die Angaben und starte das Programm erneut.')
        return None
    print('\nEingaben wurden erfolgreich validiert.')
    validated_input = validation_result.corrected_input
    game_profile = load_game_profile_from_api()
    print('\n--- Verwendetes Game-Profil ---')
    print(game_profile)
    score_result = calculate_playability_score(hardware_profile = hardware_profile, game_profile = game_profile, user_input = validated_input)
    advice = derive_graphics_advice(score_result = score_result, hardware = hardware_profile, game = game_profile)
    print('\n--- Basis-Scoring ---')
    print(f'''Score: {score_result.score}/100''')
    print(f'''Einsch├ñtzung (Scoring-Engine): {score_result.level}''')
    print(f'''Details: {score_result.details}''')
    print('\n=== Empfohlene Grafikeinstellungen ===')
    print(f'''Empfohlenes Preset: {advice.preset}''')
    print(f'''Raytracing: {advice.raytracing}''')
    print(f'''Upscaling: {advice.upscaling}''')
    print(f'''V-Sync: {advice.vsync}''')
    if advice.resolution_hint:
        print(f'''Aufl├Âsung: {advice.resolution_hint}''')
    if advice.extra_tips:
        print('\nWeitere Tipps:')
        for tip in advice.extra_tips:
            print(f'''- {tip}''')
    if advice.explanations:
        print('\nErkl├ñrungen zu den Einstellungen:')
        for expl in advice.explanations:
            print(f'''- {expl}''')
    print('\n=== Intelligenter Assistent ===')
    decision = run_assistant(score_result)
    print(f'''Finaler Score des Assistenten: {decision.final_score}/100''')
    print(f'''Gesamteinsch├ñtzung: {decision.final_label}''')
    print('\nBegr├╝ndung:')
    for line in decision.explanations:
        print(f'''- {line}''')
    print('\nEmpfehlungen:')
    for rec in decision.recommendations:
        print(f'''- {rec}''')

if __name__ == '__main__':
    main()
    return None