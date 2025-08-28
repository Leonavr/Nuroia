# app/core/profiles.py
import json
import os

PROFILES_FILE = "profiles.json"

def load_profiles() -> dict:
    """
    Loads user profiles from the profiles.json file.
    Returns an empty dictionary if the file doesn't exist or is invalid.
    """
    if not os.path.exists(PROFILES_FILE):
        return {}
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Return an empty dict if the file is corrupted or unreadable
        return {}

def save_profiles(profiles: dict):
    """
    Saves the profiles dictionary to the profiles.json file.
    """
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        # ensure_ascii=False allows saving names in different languages (e.g., Cyrillic)
        json.dump(profiles, f, indent=4, ensure_ascii=False)