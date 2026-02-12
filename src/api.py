import requests
from html import unescape
import re


# URLs für die Steam-APIs zur Suche und Detailabfrage
SEARCH_URL = "https://steamcommunity.com/actions/SearchApps"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"


def search_steam_games(query: str, limit: int = 5):
    """
    Sucht Spiele auf Steam anhand eines Suchbegriffs und gibt eine Liste von Treffern zurück.
    """
    url = f"{SEARCH_URL}/{query}"  # API-Endpunkt mit Suchbegriff
    resp = requests.get(url, timeout=10)  # GET-Anfrage mit Timeout
    resp.raise_for_status()  # Fehler werfen, wenn die Anfrage fehlschlägt

    data = resp.json()  # Antwort als JSON parsen
    results = []

    for item in data[:limit]:  # Nur die ersten 'limit' Ergebnisse verarbeiten
        results.append({
            "name": item.get("name"),      # Name des Spiels
            "appid": int(item.get("appid")),  # App-ID des Spiels
        })

    return results  # Liste der gefundenen Spiele


def _clean_html(text: str) -> str:
    """
    Entfernt HTML-Tags und wandelt HTML-Entities in lesbaren Text um.
    """
    if not text:
        return ""  # Bei leerem Text sofort zurückgeben

    text = unescape(text)  # HTML-Entities wie &amp; in & umwandeln
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)  # <br> durch Zeilenumbruch ersetzen
    text = re.sub(r"<.*?>", "", text)  # Alle weiteren HTML-Tags entfernen
    return text.strip()  # Leerzeichen am Anfang und Ende entfernen


def fetch_steam_requirements(appid: int):
    """
    Lädt die Hardwareanforderungen eines Spiels von Steam ab.
    """
    params = {
        "appids": appid,  # App-ID des Spiels
        "l": "english"    # Sprache der Anforderungen (englisch)
    }

    resp = requests.get(DETAILS_URL, params=params, timeout=10)  # GET-Anfrage mit Parametern
    resp.raise_for_status()  # Fehler werfen, wenn die Anfrage fehlschlägt

    data = resp.json()  # Antwort als JSON parsen
    app_data = data.get(str(appid), {}).get("data")  # Daten des Spiels extrahieren

    if not app_data:
        return None  # Keine Daten gefunden

    pc_reqs = app_data.get("pc_requirements", {})  # PC-Anforderungen extrahieren

    return {
        "minimum": _clean_html(pc_reqs.get("minimum", "")),      # Mindestanforderungen
        "recommended": _clean_html(pc_reqs.get("recommended", "")),  # Empfohlene Anforderungen
    }
