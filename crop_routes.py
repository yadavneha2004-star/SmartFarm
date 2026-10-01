from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
from database.db import db
from models.recommendation_model import Recommendation
from services.weather_service import get_seasonal_weather
from services.market_service import get_market_intelligence
from services.pest_service import detect_pest_risk
from services.location_service import analyze_regional_suitability
from services.schemes_service import get_all_schemes

crop_bp = Blueprint('crop_routes', __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CROP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'crop_model.pkl')
CROP_DETAILS_PATH = os.path.join(PROJECT_ROOT, 'crop_details.json')

try:
    crop_model = joblib.load(CROP_MODEL_PATH)
    with open(CROP_DETAILS_PATH, 'r') as f:
        crop_details_db = json.load(f)
except Exception as e:
    crop_model = None
    crop_details_db = {}
    print(f"Failed to load models in crop_routes: {e}")

@crop_bp.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    if not data: return jsonify({"error": "Missing input data"}), 400

    city = data.get("city")
    if not city: return jsonify({"error": "City name is required"}), 400

    user_id = data.get("user_id") # Optional

    # 1. Weather
    weather, error = get_seasonal_weather(city)
    if error: return jsonify({"error": error}), 503

    try:
        # 2. AI Prediction
        feat_dict = {
            'N': float(data.get('N', 0)), 'P': float(data.get('P', 0)), 'K': float(data.get('K', 0)),
            'temperature': weather['avg_temp'], 'humidity': weather['avg_humidity'],
            'ph': float(data.get('ph', 6.5)), 'rainfall': weather['total_rainfall']
        }
        features_df = pd.DataFrame([feat_dict])
        prediction = crop_model.predict(features_df)[0]
        confidence = float(np.max(crop_model.predict_proba(features_df)[0]))
        crop_name = str(prediction).capitalize()

        # 3. Market Intelligence Integration
        state = weather.get('region', 'Karnataka')
        land_size = float(data.get('farm_size', 1))
        market_intel, _ = get_market_intelligence(crop_name, state, yield_tons=4 * land_size)

        # 4. Pest Risk
        pest_alerts = detect_pest_risk(weather['avg_temp'], weather['avg_humidity'])

        # 5. Schemes
        schemes = get_all_schemes()

        # 6. Response Construction
        response = {
            "recommended_crop": crop_name,
            "confidence": f"{confidence:.0%}",
            "weather_summary": weather,
            "market_price": market_intel,
            "pest_alerts": pest_alerts,
            "regional_insight": analyze_regional_suitability(city, crop_name),
            "government_schemes": schemes,
            "crop_details": crop_details_db.get(crop_name, {})
        }

        # 7. Save to DB if user_id is provided
        if user_id:
            rec = Recommendation(
                user_id=user_id,
                recommended_crop=crop_name,
                confidence=f"{confidence:.0%}",
                soil_data=data,
                weather_data=weather
            )
            db.session.add(rec)
            db.session.commit()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Advisory Pipeline Error: {str(e)}"}), 500
