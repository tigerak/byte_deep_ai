from flask import Flask
from flask_cors import CORS
# modules
import config as config


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    
    # CORS 설정 추가
    # 모든 도메인에서 /api/* 경로로의 요청을 허용하도록 설정합니다. 특정 도메인만 허용하려면 {"origins": "http://127.0.0.1:5000"}와 같이 설정할 수 있습니다.
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    from app.main import bp
    app.register_blueprint(bp)
    
    return app

app = create_app()
    
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)