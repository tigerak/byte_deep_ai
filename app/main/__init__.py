from flask import Blueprint
# Modules
from config import *

# bp = Blueprint('main', __name__, url_prefix='/')
bp = Blueprint('main', __name__)

from app.main import routes