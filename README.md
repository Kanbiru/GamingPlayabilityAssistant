# Gaming Playability Assistant
**Intelligenter Assistent zur Bewertung der PC-Spielbarkeit**

## Beschreibung
Das Gaming Playability Assistant analysiert, ob ein PC-Spiel auf deiner Hardware flüssig läuft und empfiehlt optimale Grafikeinstellungen. Es kombiniert:

- Automatische **Hardware-Erkennung** (CPU, GPU, RAM, WinSAT)
- **Steam-API** für Spieleanforderungen
- **Regelbasierte + KI-Bewertung** (Playability Score + neuronales Netz)
- Konkrete **Grafikeinstellungsempfehlungen** (Preset, Raytracing, Upscaling)


## Voraussetzungen
- **Windows 10/11** (WinSAT erforderlich)
- **Python 3.10+**
- Internetverbindung (Steam-API)


## requirements.txt
requests==2.31.0
torch==2.3.0
psutil==5.9.8
numpy==1.26.0
pandas==2.1.0
scikit-learn==1.5.0


**Versionierung ist konservativ/stabil gewählt** (Dezember 2025-Stand):
- ✅ `requests 2.31.0` — bewährte Version, keine Breaking Changes
- ✅ `torch 2.3.0` — stabil für dein NN, CPU-only
- ✅ `psutil 5.9.8` — bewährt für Windows Systemabfragen
- ✅ `numpy 1.26.0` — kompatibel mit torch/pandas
- ✅ `pandas 2.1.0` — stabil, wahrscheinlich für interne Datenverarbeitung
- ✅ `scikit-learn 1.5.0` — wahrscheinlich für Scoring-Features oder Metriken


## Schnellstart

```bash
# 1. Repository klonen
git clone <dein-repo-url>
cd gaming-playability-assistant


# 2. Abhängigkeiten installieren
pip install -r requirements.txt


# 3. Programm starten
python ui.py