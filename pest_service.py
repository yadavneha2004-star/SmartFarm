import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# server.py is in backend/, this file is in backend/services/
# So PROJECT_ROOT should be the root folder
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
PEST_RULES_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'pest_rules.json')

def detect_pest_risk(temperature, humidity):
    """
    Analyzes weather data against pest rules to identify risks.
    """
    if not os.path.exists(PEST_RULES_PATH):
        # Try fallback to project root if called differently
        ALT_PATH = os.path.join(os.getcwd(), 'datasets', 'pest_rules.json')
        if os.path.exists(ALT_PATH):
            rules_path = ALT_PATH
        else:
            return []
    else:
        rules_path = PEST_RULES_PATH

    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        
        risks = []
        for pest, conditions in rules.items():
            h_min = conditions.get("humidity_min", 100)
            t_min = conditions.get("temperature_min", 0)
            t_max = conditions.get("temperature_max", 100)
            
            # Simple Threshold Matching
            if humidity >= h_min and t_min <= temperature <= t_max:
                risks.append({
                    "pest": pest.replace("_", " ").capitalize(),
                    "risk_level": "High",
                    "recommendation": conditions.get("recommendation", "Consult local expert")
                })
        
        return risks
    except Exception as e:
        print(f"Pest Service Error: {e}")
        return []
