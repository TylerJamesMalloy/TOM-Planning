from xmlrpc.client import Boolean
from pettingzoo.tom import liars_dice_v0, commune_v0
from pettingzoo.classic import mahjong_v4, texas_holdem_v4, chess_v4, tictactoe_v3, connect_four_v3
from tom.models.madqn import MADQN
from tom.models.mbdqn import MBDQN
from tom.models.matom import MATOM
from typing import Any, Dict, List
import numpy as np
import torch as th 
import argparse
import time 
import os
import re 

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# print parser args to file to load 
parser = argparse.ArgumentParser(description='Options for Multi-Agent Theory of Mind Model.')
parser.add_argument('--model_type', default = 'MADQN', type=str, help='matom, madqn, mbdqn')
parser.add_argument('--player_id', default = 0, type=int, help='ID of player for compare runs')
parser.add_argument('--timesteps',  default = 10000, type=int, help='Number of timesteps to run')
parser.add_argument('--log_time', default = 1000, type=int, help="Frequency to log model")
parser.add_argument('--layers', default=[64,64], type=List,  help='Layer structure for NNs')
parser.add_argument('--compare', default=False, dest='compare', action='store_true')
parser.add_argument('--compare_folder', default='./', type=str,  help='Model to save model comparison results')
parser.add_argument('--load_folder', default='./', type=str,  help='Model to save folders')
parser.add_argument('--save_folder', default='./trained_models/models/test/', type=str,  help='Model to load folders')
parser.add_argument('--opponent', default='random', type=str,  help='Model to load opponent from or string opponent type')
parser.add_argument('--batch_size', default=64, type=int,  help='Model to save folders')
parser.add_argument('--device', default="cuda", type=str,  help='Model to save folders')
parser.add_argument('--gamma', default=0.99, type=int,  help='Model to save folders')
parser.add_argument('--target_update', default=10, type=int,  help='Target update network frequency')
parser.add_argument('--model_based', default=False, dest='model_based', action='store_true')
parser.add_argument('--planning_depth', default=1000, type=int, help="number of samples to run")
parser.add_argument('--base_model', default='', type=str,  help='Base model for planning')
parser.add_argument('--learning_rate', default=1e-4, type=float,  help='Base model for planning')

parser.add_argument('--attention_size', default=10, type=float,  help='Base model for planning')

args = parser.parse_args()

env = tictactoe_v3.env()

# python train

print("training model, ", args.model_type)

if(args.model_type == "MATOM"):
    model = MATOM(env=env, args=args)
elif(args.model_type == "MADQN"):
    model = MADQN(env=env, args=args)
elif(args.model_type == "MBDQN"):
    model = MBDQN(env=env, args=args)
else:
    assert(False)

if(args.compare):
    data = model.compare(args)
    opponent_tag = args.opponent if args.opponent == 'random' else re.split('/', args.opponent)[-2]

    path = os.path.abspath(os.path.join(args.compare_folder , args.load_folder + "p" + str(args.player_id) , opponent_tag))
    if(not os.path.exists(path)): os.makedirs(path)
    np.save(path +  "/results.npy", data)
else:
    data = model.learn(total_timesteps = args.timesteps)
    #model.save(args.save_folder, "models-" + str(int(args.timesteps / args.log_time)))
    data.to_pickle(args.save_folder + "/results.pkl")


