import os 


ts = " 100000 "
pid = " 0 "
compare_folder = " ./test_results/no_limit_holdem/8_players/TOM/1M_ts/models-17/"
load_folder = " ./trained_models/no_limit_holdem/8_players/TOM/1M_ts/models-17/"
opponent = " random "
num_players = " 8 "
model_type = " TOM "
environment = " no_limit_holdem "
p_depth = " 100 "

command = "python train.py --compare  "
command += "--planning_depth " + p_depth
command += " --player_id " + pid 
command += " --timesteps  " + ts 
command += " --compare_folder " + compare_folder 
command += " --load_folder " + load_folder
command += " --opponent " + opponent
command += " --num_players " + num_players
command += " --model_type " + model_type
command += " --environment " + environment
print(command)

os.system(command)



