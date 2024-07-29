from flask import Blueprint

# bp = Blueprint('main', __name__, url_prefix='/')
bp = Blueprint('main', __name__)

from app.main import routes