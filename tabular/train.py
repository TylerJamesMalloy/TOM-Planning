import numpy as np 
import argparse

from envs.abstract import AbstractEnv
from models.base import Base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for Tabular Multi-Agent Theory of Mind Model.')
    parser.add_argument('--num_players', default = 3, type=int, help='Number of players')
    parser.add_argument('--num_objects', default = 3, type=int, help='Number of observed objects')
    parser.add_argument('--num_actions', default = 3, type=int, help='Number of possible actions')
    parser.add_argument('--obj_states', default = 3, type=int, help='Number of types of objects')
    parser.add_argument('--episode_length', default = 10, type=int, help='Length of episode in timesteps')

    parser.add_argument('--timesteps', default = 100, type=int, help='Training timesteps')
    parser.add_argument('--seed', default = 12345, type=int, help='Training timesteps')
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = AbstractEnv(args)
    model = Base(env, args)

    results = model.learn(1000)
    print(results)