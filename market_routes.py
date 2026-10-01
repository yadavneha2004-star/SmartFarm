from flask import Blueprint, request, jsonify
from services.market_service import get_market_prices, get_best_market

market_bp = Blueprint('market_routes', __name__)

@market_bp.route('/market_prices', methods=['POST'])
def market_prices():
    data = request.get_json()
    if not data or 'commodity' not in data:
        return jsonify({"error": "Commodity is required"}), 400
    
    commodity = data['commodity']
    state = data.get('state')
    
    records, error = get_market_prices(commodity, state)
    if error and not records:
        return jsonify({"error": error}), 503
    
    best = get_best_market(records)
    
    return jsonify({
        "commodity": commodity.capitalize(),
        "best_market": best,
        "market_comparison": records
    }), 200
