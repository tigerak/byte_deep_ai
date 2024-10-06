# Flask
from flask import request, jsonify
import requests
# modules
from config import *
from app.main import bp
# from app.main.task import process_inference
from function.utils.sft_inference import SFT_inference


print("Loading model in worker...")
sft_inf = SFT_inference(model_name=MODEL_NAME)
print("Model loaded in worker.")

    
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
    