"""
@Time ： 9/1/2024 12:43 AM
@Auth ： Yizhi Hao
@File ：app
@IDE ：PyCharm
"""

from flask import Flask
from viz_v2.backend.routes.zoning_routes import zoning_bp
from viz_v2.backend.config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Register Blueprints
app.register_blueprint(zoning_bp, url_prefix='/api')

if __name__ == "__main__":
    app.run(debug=True)