from flask import Flask

def create_app():
    app = Flask(__name__)

    from bot.controller import bot_controller

    app.register_blueprint(bot_controller.bp)

    return app