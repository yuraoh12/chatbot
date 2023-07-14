from flask import Flask

def create_app():
    app = Flask(__name__)

    from mybot.views import iris_views
    app.register_blueprint(iris_views.bp)

    return app