from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
import io
import joblib
import pandas as pd
import gc
import json
import logging
import os
import threading
import traceback

# --- TensorFlow Memory Optimization ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU') # Force CPU to save driver overhead
except Exception:
    pass

# Services
from services.weather_service import get_seasonal_weather, get_weather_data
from services.market_service import get_market_prices, get_best_market, get_market_intelligence
from services.pest_service import detect_pest_risk
from services.location_service import analyze_regional_suitability
from services.schemes_service import get_all_schemes

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("agroverse-api")

# --- Configuration ---
API_KEY_SECRET = os.getenv("AGROVERSE_API_SECRET", "myAgroversePrivateKey2026")


def require_api_key(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if api_key and api_key == API_KEY_SECRET:
            return f(*args, **kwargs)
        return jsonify({"status": "error", "message": "Unauthorized: Invalid API Key"}), 401

    return decorated


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info("Current working directory: %s", os.getcwd())
logger.info("Backend base directory: %s", BASE_DIR)

SOIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "soil_model.h5")
PLANT_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_model.keras")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")

SOIL_LABELS_PATH = os.path.join(BASE_DIR, "models", "soil_labels.txt")
PLANT_LABELS_PATH = os.path.join(BASE_DIR, "models", "plant_labels.txt")
CROP_DETAILS_PATH = os.path.join(BASE_DIR, "..", "crop_details.json")
FARM_LOG_PATH = os.path.join(BASE_DIR, "database", "farm_log.json")
COMMUNITY_PATH = os.path.join(BASE_DIR, "database", "community.json")

# --- Disease Treatment Database ---
TREATMENT_DB = {
    "Apple___Apple_scab": "Apply fungicides like captan or sulfur. Remove fallen leaves in autumn.",
    "Apple___Black_rot": "Prune out dead or diseased branches. Apply copper-based fungicides.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper trees if possible. Apply fungicides at bud break.",
    "Apple___healthy": "No treatment needed. Continue regular monitoring and pruning.",
    "Blueberry___healthy": "No treatment needed. Maintain acidic soil pH.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur or potassium bicarbonate fungicides. Improve air circulation.",
    "Cherry_(including_sour)___healthy": "No treatment needed. Keep area free of debris.",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Use resistant hybrids. Apply foliar fungicides if disease pressure is high.",
    "Corn_(maize)___Common_rust": "Plant resistant varieties. Fungicides are rarely needed but available.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and use resistant hybrids. Apply fungicides if needed.",
    "Corn_(maize)___healthy": "No treatment needed. Ensure proper fertilization.",
    "Grape___Black_rot": "Prune vines to improve airflow. Apply fungicides from bud break to bloom.",
    "Grape___Esca_(Black_Measles)": "Prune out infected wood. Use wound protectants during pruning.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper fungicides. Remove infected leaves.",
    "Grape___healthy": "No treatment needed. Prune annually for vine health.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control Asian citrus psyllid. Remove infected trees to prevent spread.",
    "Peach___Bacterial_spot": "Plant resistant varieties. Apply copper sprays during dormancy.",
    "Peach___healthy": "No treatment needed. Thin fruit for better size and health.",
    "Pepper,_bell___Bacterial_spot": "Use pathogen-free seeds. Apply copper-based bactericides.",
    "Pepper,_bell___healthy": "No treatment needed. Maintain consistent watering.",
    "Potato___Early_blight": "Rotate crops. Apply fungicides like chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Use certified seed potatoes. Apply fungicides regularly during wet weather.",
    "Potato___healthy": "No treatment needed. Hill potatoes to protect tubers.",
    "Raspberry___healthy": "No treatment needed. Prune floricanes after harvest.",
    "Soybean___healthy": "No treatment needed. Monitor for soybean aphid.",
    "Squash___Powdery_mildew": "Improve air circulation. Apply fungicides like neem oil or sulfur.",
    "Strawberry___Leaf_scorch": "Plant resistant cultivars. Remove older infected leaves.",
    "Strawberry___healthy": "No treatment needed. Mulch with straw to keep berries clean.",
    "Tomato___Bacterial_spot": "Use copper-based sprays. Avoid overhead irrigation.",
    "Tomato___Early_blight": "Apply fungicides. Rotate crops and remove lower leaves.",
    "Tomato___Late_blight": "Apply fungicides like copper or chlorothalonil. Destroy infected plants.",
    "Tomato___Leaf_Mold": "Improve greenhouse ventilation. Apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Apply fungicides. Mulch to prevent splash-back.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Apply insecticidal soap or neem oil. Increase humidity.",
    "Tomato___Target_Spot": "Apply fungicides. Ensure proper spacing for airflow.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticides or reflective mulches.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Avoid handling plants after using tobacco.",
    "Tomato___healthy": "No treatment needed. Support plants with cages or stakes.",
}

# --- Global Resources ---
models = {}
db = {}
resource_errors = {}
image_model_lock = threading.Lock()


def _load_json_file(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_label_file(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def _json_success(**payload):
    return jsonify({"status": "success", **payload})


def _json_error(message, http_status=500, **payload):
    return jsonify({"status": "error", "message": message, **payload}), http_status


def _get_crop_details(crop_name):
    crop_key = str(crop_name).strip().lower()
    crop_display = crop_key.capitalize()
    return (
        db["crops"].get(crop_display)
        or db["crops"].get(crop_key)
        or db["crops"].get(crop_key.title())
        or {}
    )


def _build_crop_options(crop_model, features_df, limit=5):
    if not hasattr(crop_model, "predict_proba") or not hasattr(crop_model, "classes_"):
        crop_name = str(crop_model.predict(features_df)[0]).strip().lower()
        return [
            {
                "crop": crop_name.capitalize(),
                "confidence": "N/A",
                "score": None,
                "details_available": bool(_get_crop_details(crop_name)),
            }
        ]

    probabilities = crop_model.predict_proba(features_df)[0]
    ranked_indexes = np.argsort(probabilities)[::-1][:limit]
    options = []
    for index in ranked_indexes:
        crop_key = str(crop_model.classes_[index]).strip().lower()
        confidence = float(probabilities[index])
        details = _get_crop_details(crop_key)
        options.append(
            {
                "crop": crop_key.capitalize(),
                "confidence": f"{confidence:.0%}",
                "score": round(confidence, 4),
                "details_available": bool(details),
                "duration": details.get("duration"),
                "soil_ph": details.get("soil_ph"),
                "season": details.get("suitable_season", []),
                "water_need": details.get("water_need", []),
            }
        )
    return options


def _validate_file(path, resource_name):
    if not os.path.exists(path):
        message = f"{resource_name} file not found: {path}"
        logger.error(message)
        raise FileNotFoundError(message)
    if os.path.getsize(path) == 0:
        message = f"{resource_name} file is empty: {path}"
        logger.error(message)
        raise FileNotFoundError(message)
    logger.info("%s file found: %s (%s bytes)", resource_name, path, os.path.getsize(path))


def _ensure_static_data():
    if "soil_labels" not in db:
        _validate_file(SOIL_LABELS_PATH, "Soil labels")
        db["soil_labels"] = _load_label_file(SOIL_LABELS_PATH)
    if "plant_labels" not in db:
        _validate_file(PLANT_LABELS_PATH, "Plant labels")
        db["plant_labels"] = _load_label_file(PLANT_LABELS_PATH)
    if "crops" not in db:
        _validate_file(CROP_DETAILS_PATH, "Crop details")
        db["crops"] = _load_json_file(CROP_DETAILS_PATH)


def _release_image_models_except(active_model_name):
    released = []
    # Forcefully clear everything from memory if we are switching
    for loaded_model_name in ("soil", "plant"):
        if loaded_model_name != active_model_name and loaded_model_name in models:
            released.append(loaded_model_name)
            del models[loaded_model_name]

    if released:
        logger.info("Released TensorFlow model(s) to save memory: %s", ", ".join(released))
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        gc.collect() # Double call to ensure nested objects are cleared


def _load_model(model_name):
    if model_name in models:
        return models[model_name]

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        if model_name == "soil":
            _release_image_models_except("soil")
            _validate_file(SOIL_MODEL_PATH, "Soil model")
            logger.info("Loading soil model from %s", SOIL_MODEL_PATH)
            models["soil"] = tf.keras.models.load_model(SOIL_MODEL_PATH, compile=False)
        elif model_name == "plant":
            _release_image_models_except("plant")
            _validate_file(PLANT_MODEL_PATH, "Plant disease model")
            logger.info("Loading plant disease model from %s", PLANT_MODEL_PATH)
            models["plant"] = tf.keras.models.load_model(PLANT_MODEL_PATH, compile=False)
        elif model_name == "crop":
            _validate_file(CROP_MODEL_PATH, "Crop model")
            logger.info("Loading crop model from %s", CROP_MODEL_PATH)
            models["crop"] = joblib.load(CROP_MODEL_PATH)
        else:
            raise ValueError(f"Unknown model requested: {model_name}")

        resource_errors.pop(model_name, None)
        logger.info("%s model loaded successfully", model_name)
        return models[model_name]
    except Exception as exc:
        resource_errors[model_name] = str(exc)
        logger.exception("Failed to load %s model", model_name)
        raise


def _model_not_ready_response(model_name, feature_name):
    return _json_error(
        f"{feature_name} is temporarily unavailable.",
        503,
        detail=resource_errors.get(model_name, "Model could not be loaded."),
    )


def _get_uploaded_image():
    logger.info(
        "Incoming %s request to %s content_type=%s files=%s",
        request.method,
        request.path,
        request.content_type,
        list(request.files.keys()),
    )
    file = request.files.get("file")
    if file is None:
        raise ValueError("No file uploaded. Send multipart/form-data with field name 'file'.")
    if not file.filename:
        raise ValueError("No file selected.")

    image_bytes = file.read()
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1024, 1024))
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid or corrupt image file.") from exc

    logger.info("Accepted image filename=%s size=%s mode=%s", file.filename, image.size, image.mode)
    return image


def _model_image_size(model, default_size):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        return int(input_shape[2]), int(input_shape[1])
    return default_size


def _prepare_image_array(image, target_size):
    image = image.resize(target_size)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def _prediction_scores(raw_prediction):
    scores = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Model returned an empty prediction.")

    if np.any(scores < 0) or not np.isclose(float(np.sum(scores)), 1.0, atol=1e-3):
        scores = tf.nn.softmax(scores).numpy()
    return scores


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return _json_error(exc.description, exc.code or 500)
    logger.exception("Unhandled server error: %s", exc)
    return _json_error("Internal server error. Check Render logs for details.", 500)


def initialize_server():
    logger.info("%s", "\n" + "=" * 50 + "\n      AgroVerse Intelligence Server Booting\n" + "=" * 50)
    try:
        _ensure_static_data()
        _load_model("crop")
        logger.info("Core advisory resources initialized.")
    except Exception as exc:
        logger.warning("Boot warning: %s", exc)


initialize_server()

# --- API Endpoints ---


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "AgroVerse Intelligence Server is running",
            "version": "1.0.0",
            "endpoints": [
                "/current_weather",
                "/market_prices",
                "/recommend_crop",
                "/predict_soil",
                "/predict_plant",
                "/schemes",
                "/farm_logs",
                "/posts",
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "agroverse-api",
            "cwd": os.getcwd(),
            "base_dir": BASE_DIR,
            "models": {
                "crop": "loaded" if "crop" in models else "unavailable",
                "plant": "loaded" if "plant" in models else "lazy",
                "soil": "loaded" if "soil" in models else "lazy",
            },
            "resource_errors": resource_errors,
        }
    )


@app.route("/current_weather", methods=["POST"])
@require_api_key
def current_weather():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    from services.weather_service import get_weather_by_coords

    lat = data.get("lat")
    lon = data.get("lon")
    city = data.get("city")

    if lat and lon:
        weather, error = get_weather_by_coords(lat, lon)
    elif city:
        weather, error = get_weather_data(city)
    else:
        return jsonify({"error": "City or Coordinates required"}), 400

    if error:
        return jsonify({"error": error}), 503
    return jsonify(weather)


@app.route("/market_prices", methods=["POST"])
@require_api_key
def market_prices():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing input data"}), 400

    commodity = data.get("commodity") or data.get("crop")
    if not commodity:
        return jsonify({"error": "Commodity or Crop is required"}), 400

    state = data.get("state")
    district = data.get("district")

    try:
        farm_size = float(data.get("farm_size", 5.0))
        yield_per_acre = float(data.get("yield_per_acre", 2.0))
    except (ValueError, TypeError):
        farm_size = 5.0
        yield_per_acre = 2.0

    records, error = get_market_prices(commodity, state, district)
    if error and not records:
        logger.warning("Market service returned no records for %s/%s/%s: %s", commodity, state, district, error)
        records = []

    best = get_best_market(records)

    estimated_revenue = 0
    if best:
        modal_price = best.get("modal_price", 0)
        estimated_revenue = round(farm_size * yield_per_acre * modal_price * 10, 2)
        best["estimated_revenue"] = estimated_revenue
        best["farm_size"] = farm_size
        best["yield_per_acre"] = yield_per_acre

    return jsonify(
        {
            "status": "success",
            "commodity": commodity.capitalize(),
            "best_market": best,
            "market_comparison": records,
            "estimated_revenue": estimated_revenue,
            "farm_size": farm_size,
            "yield_per_acre": yield_per_acre,
        }
    )


@app.route("/recommend_crop", methods=["POST"])
@require_api_key
def recommend_crop():
    data = request.get_json()
    if not data:
        return _json_error("Missing input data", 400)

    try:
        _ensure_static_data()
        crop_model = _load_model("crop")
    except Exception:
        return _model_not_ready_response("crop", "Crop recommendation")

    city = data.get("city")
    if not city:
        return _json_error("City name is required", 400)

    try:
        land_size = float(data.get("land_size", 1.0))
        if land_size <= 0:
            raise ValueError("Land size must be greater than zero.")
    except (TypeError, ValueError) as exc:
        return _json_error(f"Invalid land_size: {exc}", 400)

    weather, error = get_seasonal_weather(city)
    if error:
        return _json_error(error, 503)

    try:
        feat_dict = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "temperature": weather["avg_temp"],
            "humidity": weather["avg_humidity"],
            "ph": float(data.get("ph", 6.5)),
            "rainfall": weather["total_rainfall"],
        }
        features_df = pd.DataFrame([feat_dict])
        prediction = crop_model.predict(features_df)[0]
        rec_crop_key = str(prediction).strip().lower()
        rec_crop_name = rec_crop_key.capitalize()
        suitable_crops = _build_crop_options(crop_model, features_df)
        if suitable_crops and suitable_crops[0]["crop"]:
            rec_crop_key = suitable_crops[0]["crop"].strip().lower()
            rec_crop_name = rec_crop_key.capitalize()
        alternative_crops = [
            crop for crop in suitable_crops
            if crop["crop"].lower() != rec_crop_key
        ]

        target_crop = rec_crop_key
        target_crop_display = rec_crop_name
        state = weather.get("region", "Karnataka")

        crop_info = _get_crop_details(rec_crop_key)
        yield_per_hectare = crop_info.get("yield_per_hectare", 2.5)
        acre_to_hectare = land_size / 2.47
        total_yield_tons = round(yield_per_hectare * acre_to_hectare, 2)

        market_intel, _ = get_market_intelligence(target_crop, state, yield_tons=total_yield_tons)

        reasoning = []
        ph_value = float(data.get("ph", 6.5))
        if 6.0 <= ph_value <= 7.0:
            reasoning.append(f"Your soil pH ({ph_value}) is in the ideal neutral range.")
        if weather["avg_temp"] > 25:
            reasoning.append(f"The high average temperature ({weather['avg_temp']} C) favors {target_crop_display}.")
        if weather["total_rainfall"] > 800:
            reasoning.append(f"Abundant seasonal rainfall ({weather['total_rainfall']}mm) provides natural irrigation.")

        explanation = " ".join(reasoning) if reasoning else f"AI recommends {rec_crop_name} based on nutrient profile."
        if hasattr(crop_model, "predict_proba"):
            confidence = suitable_crops[0]["confidence"] if suitable_crops else "N/A"
        else:
            confidence = "N/A"

        response = {
            "status": "success",
            "recommended_crop": rec_crop_name,
            "target_crop_details": target_crop_display,
            "model_input": feat_dict,
            "confidence": confidence,
            "suitable_crops": suitable_crops,
            "alternative_crops": alternative_crops,
            "weather_summary": weather,
            "market_intelligence": market_intel,
            "pest_alerts": detect_pest_risk(weather["avg_temp"], weather["avg_humidity"]),
            "regional_insight": analyze_regional_suitability(city, target_crop_display),
            "crop_details": crop_info,
            "land_size_acres": land_size,
            "estimated_yield_tons": total_yield_tons,
            "explanation": explanation,
        }
        logger.info("Crop recommendation city=%s crop=%s confidence=%s fallback_weather=%s", city, rec_crop_name, confidence, weather.get("is_fallback"))
        return jsonify(response)
    except (TypeError, ValueError) as exc:
        logger.warning("Bad crop recommendation request: %s", exc)
        return _json_error(f"Invalid crop recommendation input: {str(exc)}", 400)
    except Exception as exc:
        logger.error("Advisory Pipeline Error: %s\n%s", exc, traceback.format_exc())
        return _json_error(f"Advisory Pipeline Error: {str(exc)}", 500)


@app.route("/predict_soil", methods=["POST"])
@require_api_key
def predict_soil():
    try:
        img = _get_uploaded_image()
    except ValueError as exc:
        logger.warning("Bad soil prediction request: %s", exc)
        return _json_error(str(exc), 400)

    try:
        _ensure_static_data()
        with image_model_lock:
            soil_model = _load_model("soil")
            arr = _prepare_image_array(img, _model_image_size(soil_model, (128, 128)))
            preds = soil_model.predict(arr, verbose=0)[0]
    except Exception:
        return _model_not_ready_response("soil", "Soil analysis")

    try:
        scores = _prediction_scores(preds)
        idx = int(np.argmax(scores))

        if idx >= len(db["soil_labels"]):
            raise ValueError("Soil labels are out of sync with the loaded model.")

        logger.info("Soil prediction idx=%s label=%s confidence=%.4f", idx, db["soil_labels"][idx], float(scores[idx]))
        return _json_success(
            soil_type=db["soil_labels"][idx],
            prediction=db["soil_labels"][idx],
            confidence=f"{float(scores[idx]):.1%}",
            confidence_score=round(float(scores[idx]), 4),
        )
    except ValueError as exc:
        logger.warning("Bad soil prediction request: %s", exc)
        return _json_error(str(exc), 400)
    except Exception as exc:
        logger.error("Soil Analysis Error: %s\n%s", exc, traceback.format_exc())
        return _json_error(f"Soil Analysis Error: {str(exc)}", 500)


@app.route("/predict_plant", methods=["POST"])
@require_api_key
def predict_plant():
    try:
        img = _get_uploaded_image()
    except ValueError as exc:
        logger.warning("Bad plant prediction request: %s", exc)
        return _json_error(str(exc), 400)

    try:
        _ensure_static_data()
        with image_model_lock:
            plant_model = _load_model("plant")
            arr = _prepare_image_array(img, _model_image_size(plant_model, (224, 224)))
            preds = plant_model.predict(arr, verbose=0)[0]
    except Exception:
        return _model_not_ready_response("plant", "Disease detection")

    try:
        scores = _prediction_scores(preds)
        idx = int(np.argmax(scores))

        if idx >= len(db["plant_labels"]):
            raise ValueError("Plant labels are out of sync with the loaded model.")

        disease_name = db["plant_labels"][idx]
        treatment = TREATMENT_DB.get(disease_name, "Consult an agricultural expert for specific treatment.")
        display_name = disease_name.replace("___", " ").replace("_", " ")

        logger.info(
            "Plant prediction idx=%s raw_label=%s confidence=%.4f",
            idx,
            disease_name,
            float(scores[idx]),
        )
        return _json_success(
            prediction=display_name,
            disease=disease_name,
            confidence=f"{float(scores[idx]):.1%}",
            confidence_score=round(float(scores[idx]), 4),
            treatment=treatment,
        )
    except ValueError as exc:
        logger.warning("Bad plant prediction request: %s", exc)
        return _json_error(str(exc), 400)
    except Exception as exc:
        logger.error("Disease Detection Error: %s\n%s", exc, traceback.format_exc())
        return _json_error(f"Disease Detection Error: {str(exc)}", 500)


@app.route("/schemes", methods=["GET"])
@require_api_key
def get_schemes():
    return jsonify(get_all_schemes())


@app.route("/farm_logs", methods=["GET"])
@require_api_key
def get_farm_logs():
    if not os.path.exists(FARM_LOG_PATH):
        return jsonify([])
    with open(FARM_LOG_PATH, "r", encoding="utf-8") as file_obj:
        return jsonify(json.load(file_obj))


@app.route("/add_farm_log", methods=["POST"])
@require_api_key
def add_farm_log():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    logs = []
    if os.path.exists(FARM_LOG_PATH):
        with open(FARM_LOG_PATH, "r", encoding="utf-8") as file_obj:
            logs = json.load(file_obj)
    logs.append(
        {
            "date": data.get("date", ""),
            "activity": data.get("activity", ""),
            "expense": data.get("expense", 0),
            "crop_stage": data.get("crop_stage", ""),
            "notes": data.get("notes", ""),
        }
    )
    with open(FARM_LOG_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(logs, file_obj)
    return jsonify({"success": True})


@app.route("/posts", methods=["GET"])
@require_api_key
def get_posts():
    if not os.path.exists(COMMUNITY_PATH):
        return jsonify([])
    with open(COMMUNITY_PATH, "r", encoding="utf-8") as file_obj:
        return jsonify(json.load(file_obj))


@app.route("/add_post", methods=["POST"])
@require_api_key
def add_post():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    posts = []
    if os.path.exists(COMMUNITY_PATH):
        with open(COMMUNITY_PATH, "r", encoding="utf-8") as file_obj:
            posts = json.load(file_obj)
    posts.append(
        {
            "id": len(posts) + 1,
            "author": data.get("author", "Farmer"),
            "title": data.get("title", ""),
            "content": data.get("content", ""),
            "replies": [],
        }
    )
    with open(COMMUNITY_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(posts, file_obj)
    return jsonify({"success": True})


@app.route("/add_reply", methods=["POST"])
@require_api_key
def add_reply():
    data = request.get_json()
    post_id = data.get("post_id")
    if not post_id:
        return jsonify({"error": "Missing post_id"}), 400
    posts = []
    if os.path.exists(COMMUNITY_PATH):
        with open(COMMUNITY_PATH, "r", encoding="utf-8") as file_obj:
            posts = json.load(file_obj)
    for post in posts:
        if post["id"] == post_id:
            post["replies"].append(
                {
                    "author": data.get("author", "Farmer"),
                    "content": data.get("content", ""),
                }
            )
            break
    with open(COMMUNITY_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(posts, file_obj)
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
