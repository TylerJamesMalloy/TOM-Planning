import os 


for model_id in range(0,101):
    print("model id: ", str(model_id))
    for player_id in range(0,2):
        call = "python train.py --compare --model_type MADQN --load_folder ./trained_models/tiktaktoe/MADQN_10K/models-" + str(model_id) + "/ --opponent ./trained_models/tiktaktoe/MADQN_10K/models-100/ --timesteps 100 " + " --player_id " + str(player_id)
        os.system(call)

        call = "python train.py --compare --model_type MADQN --load_folder ./trained_models/tiktaktoe/MADQN_10K/models-" + str(model_id) + "/ --opponent random --timesteps 100  " + " --player_id " + str(player_id)
        os.system(call)
