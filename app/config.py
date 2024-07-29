import sys
# modules

from secret import *

sys.path.append(BASE_DIR)

# secret key for CSRF token
SECRET_KEY = csrf_token_secret

MODEL_NAME = model_name
