import os 


for model_id in range(1,101):
    for player_id in range(0,4):
        print("model id: ", str(model_id), " player id: ", player_id)
        os.system("python train.py --compare --load_folder ./trained_models/liarsdice/DQN_1H/models-" + str(model_id) + "/ --opponent ./trained_models/liarsdice/DQN_1M/models-10 --timesteps 1000 --compare_folder ./results --player_id " + str(player_id))


# python train.py --compare --load_folder ./trained_models/liarsdice/DQN_100K/models-1/ --opponent ./trained_models/liarsdice/DQN_1M/models-10 --timesteps 1000 --compare_folder ./results --player_id 0