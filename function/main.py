
import sys
BASE_DIR=r"/home/deep_ai/Project/"
sys.path.append(BASE_DIR)

import argparse
# torch
import torch
# modules
from utils.sft_tarin import SFT_train
from utils.sft_inference import SFT_inference

def mode_train():
    pass

def mode_inference():
    pass

if __name__ == '__main__':
    model_name = 'solar_recovery'

    parser = argparse.ArgumentParser(description="Run training or inference mode.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help="Mode to run: train or inference")
    parser.add_argument('--work', type=str, choices=['merge', 'rt', ''], help="Additional option for work")
    parser.add_argument('--add', type=str, choices=['add', ''], help="Mode to train: add or none")
    parser.add_argument('--date', type=str, required=True, help="Current date in the format yy_mm_dd")
    parser.add_argument('--time', type=str, required=True, help="Current time in the format HH_MM_SS")
    
    args = parser.parse_args()
    
    date = args.date
    time = args.time

    if args.mode == 'train':
        sft_train = SFT_train(model_name)
        if args.add == 'add':
            sft_train.train(date, time, add_training=True)
        else:
            sft_train.train(date, time)
    elif args.mode == 'inference':
        sft_inf = SFT_inference(model_name)
        if args.work == 'merge':
            sft_inf.merge()
        elif args.work == 'rt':
            sft_inf.tensor_rt()
        else:
            data={}
            sft_inf.inference(data)