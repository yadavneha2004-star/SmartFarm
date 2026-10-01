import numpy as np
from sklearn.linear_model import LinearRegression

def predict_price_trend(price_history):
    """
    Predicts the next 7 days of prices based on historical trends.
    Using Linear Regression for trend analysis.
    """
    if not price_history or len(price_history) < 2:
        return {
            "predicted_prices": [],
            "trend": "neutral",
            "recommendation": "Insufficient data for prediction"
        }

    # Prepare data for Linear Regression
    # X = days (0, 1, 2, ...), y = prices
    X = np.array(range(len(price_history))).reshape(-1, 1)
    y = np.array(price_history)

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 7 days
    future_days = np.array(range(len(price_history), len(price_history) + 7)).reshape(-1, 1)
    predictions = model.predict(future_days)
    predictions = [round(float(p), 2) for p in predictions]

    # Determine trend
    current_price = price_history[-1]
    next_week_price = predictions[-1]
    
    price_change_pct = ((next_week_price - current_price) / current_price) * 100

    if price_change_pct > 2:
        trend = "upward"
        recommendation = "Hold crop for 5–7 days for better prices."
    elif price_change_pct < -2:
        trend = "downward"
        recommendation = "Sell now! Prices are expected to drop."
    else:
        trend = "stable"
        recommendation = "Prices are stable. Sell as per your convenience."

    return {
        "predicted_prices": predictions,
        "next_week_price": round(next_week_price, 2),
        "trend": trend,
        "recommendation": recommendation,
        "confidence_score": "High (Based on local trend)"
    }
