import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEME_PATHS = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "datasets", "farming_schemes.json")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "farming_schemes.json")),
    os.path.abspath(os.path.join(os.getcwd(), "datasets", "farming_schemes.json")),
]

def get_all_schemes():
    for scheme_path in SCHEME_PATHS:
        if not os.path.exists(scheme_path):
            continue
        try:
            with open(scheme_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Schemes Service Error: {e}")
            return []

    print("Warning: Schemes file not found. Returning empty list.")
    return []

def get_schemes_by_category(category):
    schemes = get_all_schemes()
    return [s for s in schemes if s.get('category') == category]
