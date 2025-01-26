from flask import Flask
from .server.main import main_routes

def create_app():
    app = Flask(__name__)

    app.config.from_object('app.config.Config')

    app.register_blueprint(main_routes)

    return app