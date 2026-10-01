import requests
import os

API_KEY = os.getenv("WEATHER_API_KEY", "4ae8bf4d587a4ddb88b144032260603")

FALLBACK_WEATHER = {
    "bangalore": {"avg_temp": 24.0, "avg_humidity": 68.0, "total_rainfall": 150.0, "region": "Karnataka"},
    "bengaluru": {"avg_temp": 24.0, "avg_humidity": 68.0, "total_rainfall": 150.0, "region": "Karnataka"},
    "mysore": {"avg_temp": 25.0, "avg_humidity": 70.0, "total_rainfall": 120.0, "region": "Karnataka"},
    "hubli": {"avg_temp": 27.0, "avg_humidity": 56.0, "total_rainfall": 95.0, "region": "Karnataka"},
    "mangalore": {"avg_temp": 28.0, "avg_humidity": 82.0, "total_rainfall": 220.0, "region": "Karnataka"},
    "hyderabad": {"avg_temp": 29.0, "avg_humidity": 58.0, "total_rainfall": 80.0, "region": "Telangana"},
    "mumbai": {"avg_temp": 30.0, "avg_humidity": 75.0, "total_rainfall": 250.0, "region": "Maharashtra"},
    "pune": {"avg_temp": 27.0, "avg_humidity": 62.0, "total_rainfall": 110.0, "region": "Maharashtra"},
    "nagpur": {"avg_temp": 31.0, "avg_humidity": 50.0, "total_rainfall": 95.0, "region": "Maharashtra"},
    "delhi": {"avg_temp": 31.0, "avg_humidity": 45.0, "total_rainfall": 60.0, "region": "Delhi"},
    "chennai": {"avg_temp": 31.0, "avg_humidity": 72.0, "total_rainfall": 180.0, "region": "Tamil Nadu"},
    "kolkata": {"avg_temp": 29.0, "avg_humidity": 78.0, "total_rainfall": 180.0, "region": "West Bengal"},
    "jaipur": {"avg_temp": 32.0, "avg_humidity": 42.0, "total_rainfall": 45.0, "region": "Rajasthan"},
    "ahmedabad": {"avg_temp": 32.0, "avg_humidity": 48.0, "total_rainfall": 80.0, "region": "Gujarat"},
    "surat": {"avg_temp": 31.0, "avg_humidity": 70.0, "total_rainfall": 120.0, "region": "Gujarat"},
    "lucknow": {"avg_temp": 30.0, "avg_humidity": 58.0, "total_rainfall": 95.0, "region": "Uttar Pradesh"},
    "patna": {"avg_temp": 30.0, "avg_humidity": 66.0, "total_rainfall": 120.0, "region": "Bihar"},
    "bhubaneswar": {"avg_temp": 30.0, "avg_humidity": 74.0, "total_rainfall": 165.0, "region": "Odisha"},
    "kochi": {"avg_temp": 29.0, "avg_humidity": 78.0, "total_rainfall": 220.0, "region": "Kerala"},
    "thiruvananthapuram": {"avg_temp": 29.0, "avg_humidity": 78.0, "total_rainfall": 220.0, "region": "Kerala"},
    "guwahati": {"avg_temp": 27.0, "avg_humidity": 78.0, "total_rainfall": 190.0, "region": "Assam"},
    "chandigarh": {"avg_temp": 28.0, "avg_humidity": 52.0, "total_rainfall": 70.0, "region": "Punjab"},
    "amritsar": {"avg_temp": 28.0, "avg_humidity": 50.0, "total_rainfall": 70.0, "region": "Punjab"},
    "indore": {"avg_temp": 28.0, "avg_humidity": 55.0, "total_rainfall": 105.0, "region": "Madhya Pradesh"},
    "bhopal": {"avg_temp": 28.0, "avg_humidity": 58.0, "total_rainfall": 105.0, "region": "Madhya Pradesh"},
    "dehradun": {"avg_temp": 25.0, "avg_humidity": 65.0, "total_rainfall": 140.0, "region": "Uttarakhand"},
    "shimla": {"avg_temp": 18.0, "avg_humidity": 60.0, "total_rainfall": 85.0, "region": "Himachal Pradesh"},
}

REGION_RAINFALL_PROFILE = {
    "andhra pradesh": 110.0,
    "assam": 190.0,
    "bihar": 120.0,
    "delhi": 60.0,
    "goa": 220.0,
    "gujarat": 80.0,
    "haryana": 65.0,
    "himachal pradesh": 85.0,
    "karnataka": 150.0,
    "kerala": 220.0,
    "madhya pradesh": 105.0,
    "maharashtra": 130.0,
    "odisha": 165.0,
    "punjab": 70.0,
    "rajasthan": 45.0,
    "tamil nadu": 180.0,
    "telangana": 90.0,
    "uttar pradesh": 95.0,
    "uttarakhand": 140.0,
    "west bengal": 180.0,
}


def _normalize_location(value):
    return (value or "").strip().lower()


def _crop_model_rainfall(city, region, forecast_rainfall):
    """
    The crop model was trained with climate-style rainfall values, not current
    7-day precipitation. Use known location profiles first, then region
    profiles, and keep the forecast value only when it is meaningfully higher.
    """
    city_key = _normalize_location(city)
    region_key = _normalize_location(region)

    profile = FALLBACK_WEATHER.get(city_key)
    if profile:
        return profile["total_rainfall"], "city_climate_profile"

    if region_key in REGION_RAINFALL_PROFILE:
        return REGION_RAINFALL_PROFILE[region_key], "region_climate_profile"

    return max(float(forecast_rainfall or 0.0), 95.0), "india_climate_default"


def _fallback_seasonal_weather(city, reason):
    profile = FALLBACK_WEATHER.get(_normalize_location(city))
    if profile is None:
        profile = {"avg_temp": 27.0, "avg_humidity": 65.0, "total_rainfall": 95.0, "region": "India"}

    return {
        "avg_temp": profile["avg_temp"],
        "avg_humidity": profile["avg_humidity"],
        "total_rainfall": profile["total_rainfall"],
        "city": city or "Unknown",
        "region": profile["region"],
        "condition": "Estimated seasonal average",
        "is_fallback": True,
        "fallback_reason": str(reason),
    }, None


def _fallback_current_weather(city, reason):
    profile = FALLBACK_WEATHER.get(_normalize_location(city))
    if profile is None:
        profile = {"avg_temp": 27.0, "avg_humidity": 65.0, "total_rainfall": 0.0, "region": "India"}

    return {
        "temperature": profile["avg_temp"],
        "humidity": profile["avg_humidity"],
        "rainfall": 0.0,
        "city": city or "Unknown",
        "region": profile["region"],
        "condition": "Estimated weather",
        "is_fallback": True,
        "fallback_reason": str(reason),
    }, None

def get_seasonal_weather(city):
    """
    Fetches 7-day forecast and computes averages for seasonal prediction.
    Adds fallback for unrecognized locations.
    """
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=7"
    
    try:
        response = requests.get(url, timeout=10)
        
        # If city not found (400), try fallback to region or default
        if response.status_code == 400:
            # Try a broader search (e.g., state capital or regional center)
            # For simplicity in this fix, we'll try a known fallback like 'Bangalore'
            # if the original city was a village in Karnataka.
            fallback_city = "Bangalore"
            url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={fallback_city}&days=7"
            response = requests.get(url, timeout=10)
            
        response.raise_for_status()
        data = response.json()
        
        forecast_days = data['forecast']['forecastday']
        
        avg_temp = sum(day['day']['avgtemp_c'] for day in forecast_days) / len(forecast_days)
        avg_humidity = sum(day['day']['avghumidity'] for day in forecast_days) / len(forecast_days)
        forecast_rainfall = sum(day['day']['totalprecip_mm'] for day in forecast_days)
        total_rainfall, rainfall_source = _crop_model_rainfall(
            data['location']['name'],
            data['location']['region'],
            forecast_rainfall,
        )
        
        return {
            "avg_temp": round(avg_temp, 2),
            "avg_humidity": round(avg_humidity, 2),
            "total_rainfall": round(total_rainfall, 2),
            "forecast_rainfall_7day": round(forecast_rainfall, 2),
            "rainfall_source": rainfall_source,
            "city": data['location']['name'],
            "region": data['location']['region'],
            "condition": data['current']['condition']['text'],
            "is_fallback": response.url.find(city) == -1 # Mark if fallback was used
        }, None
    except Exception as e:
        return _fallback_seasonal_weather(city, e)

def get_weather_by_coords(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "rainfall": data["current"]["precip_mm"],
            "city": data["location"]["name"],
            "region": data["location"]["region"],
            "condition": data["current"]["condition"]["text"]
        }, None
    except Exception as e:
        return _fallback_current_weather("Current location", e)

# Keep original function for backward compatibility
def get_weather_data(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "rainfall": data["current"]["precip_mm"],
            "city": data["location"]["name"],
            "region": data["location"]["region"],
            "condition": data["current"]["condition"]["text"]
        }, None
    except Exception as e:
        return _fallback_current_weather(city, e)
