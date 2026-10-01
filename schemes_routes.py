from flask import Blueprint, jsonify
from services.schemes_service import get_all_schemes

schemes_bp = Blueprint('schemes_routes', __name__)

@schemes_bp.route('/schemes', methods=['GET'])
def schemes():
    data = get_all_schemes()
    return jsonify(data), 200
