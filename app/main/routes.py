import json

from flask import request, jsonify
# modules
from app.main import bp
from app.config import *
from function.utils.sft_inference import SFT_inference

print("Loading model...")
sft_inf = SFT_inference(model_name=MODEL_NAME)
print("Model loaded.")

@bp.route('/')
def index():
    print('접근함!!')
    return jsonify(
        message="You're connected to the Data Manage Server API !!",
        contant={
            "key":"Make Somthing",
            "value":"I have No Idea :["
        }
        )
    # return "You're connected to the Main Server API !!"

@bp.route('/api', methods=['POST'])
def get_data():
    data = request.form 
        
    result_dict = sft_inf.inference(data)
    return jsonify(result_dict)
    