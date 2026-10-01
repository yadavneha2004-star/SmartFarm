import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
HISTORICAL_SUMMARY_PATH = os.path.join(PROJECT_ROOT, 'knowledge', 'district_yield_summary.json')

def analyze_regional_suitability(city, recommended_crop):
    """
    Checks if the recommended crop is historically strong in the region.
    """
    if not os.path.exists(HISTORICAL_SUMMARY_PATH):
        return None

    try:
        with open(HISTORICAL_SUMMARY_PATH, 'r') as f:
            data = json.load(f)
        
        district_data = data.get("district_data", {})
        mapping = data.get("mapping", {})
        
        dist_name = city.upper().strip()
        dist_yields = district_data.get(dist_name)
        
        if not dist_yields:
            return {"note": "No historical records for this district."}

        # Boost logic
        mapped_name = mapping.get(recommended_crop.lower(), recommended_crop.lower())
        hist_yield = dist_yields.get(mapped_name.lower())
        
        top_crop = max(dist_yields, key=dist_yields.get)
        top_yield = dist_yields[top_crop]

        result = {
            "historical_yield_in_district": hist_yield or "Low/No record",
            "regional_top_performer": top_crop.capitalize(),
            "regional_top_yield": top_yield
        }
        
        if hist_yield and hist_yield > 5:
            result["suitability"] = "High (Verified by local history)"
        else:
            result["suitability"] = "Experimental (Low local yield record)"
            
        return result

    except Exception:
        return None
