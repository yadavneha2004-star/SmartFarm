from flask import Blueprint, request, jsonify
from models.user_model import User
from database.db import db

auth_bp = Blueprint('auth_routes', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not all(k in data for k in ('name', 'phone', 'location', 'farm_size')):
        return jsonify({"error": "Missing required fields"}), 400
    
    if User.query.filter_by(phone=data['phone']).first():
        return jsonify({"error": "User with this phone already exists"}), 400
    
    new_user = User(
        name=data['name'],
        phone=data['phone'],
        location=data['location'],
        farm_size=data['farm_size']
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'phone' not in data:
        return jsonify({"error": "Phone number required"}), 400
    
    user = User.query.filter_by(phone=data['phone']).first()
    if user:
        return jsonify({"message": "Login successful", "user_id": user.id, "name": user.name}), 200
    return jsonify({"error": "User not found"}), 404
