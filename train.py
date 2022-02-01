from xmlrpc.client import Boolean
from pettingzoo.tom import liars_dice_v0, commune_v0
from pettingzoo.classic import mahjong_v4, texas_holdem_v4, chess_v4, tictactoe_v3
from tom.models.madqn import MADQN
from typing import Any, Dict, List
import numpy as np
import torch as th 
import argparse
import time 
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser(description='Options for Multi-Agent Theory of Mind Model.')
parser.add_argument('--model_type', default = 'random', type=str, help='full, attention, belief, dqn')
parser.add_argument('--player_id', default = 0, type=int, help='ID of player for compare runs')
parser.add_argument('--timesteps',  default = 1000000, type=int, help='Number of timesteps to run')
parser.add_argument('--log_time', default = 10000, type=int, help="Frequency to log model")
parser.add_argument('--layers', default=[64,64], type=List,  help='Layer structure for NNs')
parser.add_argument('--compare', default=False, dest='compare', action='store_true')
parser.add_argument('--load_folder', default='./trained_models/models/', type=str,  help='Model to save folders')
parser.add_argument('--save_folder', default='./trained_models/models/', type=str,  help='Model to load folders')
parser.add_argument('--opponent', default='./trained_models/models/', type=str,  help='Model to load opponent from or string opponent type')
parser.add_argument('--batch_size', default=256, type=int,  help='Model to save folders')
parser.add_argument('--device', default="cuda", type=str,  help='Model to save folders')
parser.add_argument('--gamma', default=0.999, type=int,  help='Model to save folders')
parser.add_argument('--target_update', default=10, type=int,  help='Target update network frequency')
parser.add_argument('--planning', default=True, type=bool,  help='Model based or model free')
parser.add_argument('--base_model', default='', type=str,  help='Base model for planning')

# python .\train.py --compare --load_folder ./trained_models/texas_holdem/DQN_10M/models-100/ --opponent ./trained_models/texas_holdem/DQN_1M/models-100/ 

args = parser.parse_args()

env = liars_dice_v0.env()

model = MADQN(env=env, args=args)

if(args.compare):
    data = model.compare(args)
else:
    data = model.learn(total_timesteps = args.timesteps)
    model.save(args.save_folder, "models-100")

print("episode reward mean: ", np.mean(np.abs(data)))
print("total episode reward: ", np.sum(data, axis=0))
data = np.array(data)
model_data = data[:, args.player_id]
model_wins = np.count_nonzero(model_data > 0)
model_losses = np.count_nonzero(model_data < 0)
print("win vs. loss percentage: ", model_wins / (model_wins + model_losses))
print("win vs. game percentage: ", model_wins / len(model_data))

# .\trained_models\models-100