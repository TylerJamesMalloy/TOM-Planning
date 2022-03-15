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
parser.add_argument('--timesteps',  default = 1000000, type=int, help='Number of timesteps to run')
parser.add_argument('--log_time', default = 10000, type=int, help="Frequency to log model")
parser.add_argument('--layers', default=[64,64], type=List,  help='Layer structure for NNs')
parser.add_argument('--compare', default=False, dest='compare', action='store_true')
parser.add_argument('--compare_folder', default='./', type=str,  help='Model to save model comparison results')
parser.add_argument('--load_folder', default='./', type=str,  help='Model to save folders')
parser.add_argument('--save_folder', default='./trained_models/models/test/', type=str,  help='Model to load folders')
parser.add_argument('--batch_size', default=64, type=int,  help='Model to save folders')
parser.add_argument('--device', default="cuda", type=str,  help='Model to save folders')
parser.add_argument('--gamma', default=0.99, type=int,  help='Model to save folders')
parser.add_argument('--target_update', default=10, type=int,  help='Target update network frequency')
parser.add_argument('--planning', default=False, dest='planning', action='store_true')
parser.add_argument('--planning_depth', default=5, type=int, help="number of samples to run")
parser.add_argument('--base_model', default='', type=str,  help='Base model for planning')
parser.add_argument('--learning_rate', default=1e-4, type=float,  help='lr for NN')
parser.add_argument('--num_players', default=2, type=int,  help='Number of players')
parser.add_argument('--environment', default="liars_dice_v0", type=str,  help='Number of players')

parser.add_argument('--attention_size', default=10, type=float,  help='Number of objects to attend to')

parser.add_argument('--opponent', default='random', type=str,  help='Model to load opponent from or string opponent type')
parser.add_argument('--opponent_planning', default=False, dest='opponent_planning', action='store_true')

args = parser.parse_args()

# python .\train.py --save_folder ./trained_models/connect_four/PDQN_1M/ --num_players 2 --model_type MBDQN --environment connect_four_v3 --planning --base_model ./trained_models/connect_four/MBDQN_1M/models-99/
# python .\train.py --save_folder ./trained_models/texas_holdem_2p/PDQN_1M/ --num_players 2 --model_type MBDQN --environment texas_holdem_v4 --planning --base_model ./trained_models/texas_holdem_2p/MBDQN_1M/models-99/
# python .\train.py --save_folder ./trained_models/texas_holdem_4p/PDQN_1M/ --num_players 4 --model_type MBDQN --environment texas_holdem_v4 --planning --base_model ./trained_models/texas_holdem_4p/MBDQN_1M/models-99/


# python .\train.py --load_folder ./trained_models/connect_four/MBDQN_1M/models-99/ --compare --opponent random --model_type MBDQN --environment connect_four_v3 --planning --timesteps 1000

if(args.environment == "connect_four_v3"):
    env = connect_four_v3.env()
elif(args.environment == "texas_holdem_v4"):
    env = texas_holdem_v4.env(num_players=args.num_players)

# python train

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


