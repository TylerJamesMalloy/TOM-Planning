from xmlrpc.client import Boolean
from pettingzoo.classic import texas_holdem_no_limit_v6, leduc_holdem_v4, chess_v5
from tom.models import TOM, DQN
from typing import Any, Dict, List
import numpy as np
import torch as th 
import argparse
import time 
import os
import re 

from rlcard.agents import RandomAgent

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# print parser args to file to load 
parser = argparse.ArgumentParser(description='Options for Multi-Agent Theory of Mind Model.')
parser.add_argument('--model_type', default = 'DQN', type=str, help='DQN, TOM, FTOM')
parser.add_argument('--player_id', default = 1, type=int, help='ID of player for compare runs')
parser.add_argument('--timesteps',  default = 10000, type=int, help='Number of timesteps to run')
parser.add_argument('--log_time', default = 100, type=int, help="Frequency to log model")
parser.add_argument('--layers', default=[64,64], type=List,  help='Layer structure for NNs')
parser.add_argument('--compare', default=False, dest='compare', action='store_true')
parser.add_argument('--compare_folder', default='./', type=str,  help='Model to save model comparison results')
parser.add_argument('--load_folder', default='./', type=str,  help='Model to save folders')
parser.add_argument('--save_folder', default='./trained_models/models/test/', type=str,  help='Model to save data')
parser.add_argument('--batch_size', default=256, type=int,  help='Model to save folders')
parser.add_argument('--device', default="cuda", type=str,  help='Model to save folders')
parser.add_argument('--gamma', default=0.99, type=int,  help='Model to save folders')
parser.add_argument('--target_update', default=10, type=int,  help='Target update network frequency')
parser.add_argument('--planning', default=False, dest='planning', action='store_true')
parser.add_argument('--planning_depth', default=10, type=int, help="number of samples to run")
parser.add_argument('--planning_pretraining', default=100, type=int,  help='Number of episodes to pretrain for planning')
parser.add_argument('--base_model', default='', type=str,  help='Base model for planning')
parser.add_argument('--learning_rate', default=1e-4, type=float,  help='lr for NN')
parser.add_argument('--num_players', default=2, type=int,  help='Number of players')
parser.add_argument('--environment', default="no_limit_holdem", type=str,  help='Number of players')

# 118, 52, 418
parser.add_argument('--belief_in', default=12, type=tuple,  help='Environment specific belief input')
parser.add_argument('--belief_out', default=3, type=tuple,  help='Environment specific belief shape')
parser.add_argument('--state_size', default=39, type=tuple,  help='Environment specific state shape after concatonating with belief predictions')

parser.add_argument('--private_size', default=3, type=tuple,  help='Environment specific state shape after concatonating with belief predictions')
parser.add_argument('--public_size', default=33, type=tuple,  help='Environment specific state shape after concatonating with belief predictions')

parser.add_argument('--opponent', default='random', type=str,  help='Model to load opponent from or string opponent type')
parser.add_argument('--opponent_planning', default=False, dest='opponent_planning', action='store_true')

args = parser.parse_args()

#  python .\train.py --save_folder ./trained_models/leduc_holdem/2_players/TOM/100K/  --model_type TOM  --environment leduc_holdem 
#  python .\train.py --save_folder ./trained_models/leduc_holdem/2_players/DQN/100K/  --model_type DQN  --environment leduc_holdem 

if(args.environment == "no_limit_holdem"):
    print("num players: ", args.num_players)
    env = texas_holdem_no_limit_v6.env(num_players=args.num_players)
    env.reset()
elif(args.environment == "leduc_holdem"):
    env = leduc_holdem_v4.env(num_players=args.num_players)
    env.reset()
elif(args.environment == "chess"):
    env = chess_v5.env()
    env.reset()
else:
    assert(False)

# python train

if(args.model_type == "DQN"):
    model = DQN(env, args)
elif(args.model_type == "TOM"):
    model = TOM(env, args)
elif(args.model_type == "FTOM"):
    model = FTOM(env, args)
elif(args.model_type == "MAMB"):
    model = MAMB(env, args)
else:
    assert(False)

if(args.compare):
    data = model.compare(args)
    opponent_tag = args.opponent if args.opponent == 'random' else re.split('/', args.opponent)[-2]

    path = os.path.abspath(os.path.join(args.compare_folder , args.load_folder + "p" + str(args.player_id) , opponent_tag))
    if(not os.path.exists(path)): os.makedirs(path)
    data = np.array(data)
    for player in range(data.shape[1]):
        print("Player ", player, " mean reward", np.mean(data[:,player]))
    np.save(path +  "/results.npy", data)
else:
    data = model.learn(total_timesteps = args.timesteps)
    model.save(args.save_folder, "models-" + str(int(args.timesteps / args.log_time)))
    data.to_pickle(args.save_folder + "/losses.pkl")
    print(data)


