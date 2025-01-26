from flask import Blueprint, request, jsonify

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/status', methods=['GET'])
def check_status():
    return jsonify({"status": "ok", "message": "Le serveur est en marche."})

@main_routes.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()

    if 'message' not in data or 'wallet_address' not in data:
        return jsonify({"status": "error", "message": "Message et adresse de wallet sont requis."}), 400

    message = data['message']
    wallet_address = data['wallet_address']

    return jsonify({
        "status": "success",
        "message": f"Question posée : {message}",
        "wallet_address": wallet_address
    })

@main_routes.route('/connect', methods=['POST'])
def connect_wallet():
    data = request.get_json()

    if 'wallet_address' not in data:
        return jsonify({"status": "error", "message": "Adresse du wallet requise."}), 400

    wallet_address = data['wallet_address']

    return jsonify({
        "status": "success",
        "message": f"Connexion réussie avec le wallet {wallet_address}."
    })