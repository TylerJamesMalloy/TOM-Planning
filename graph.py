import pandas 




for model_id in range(1,101):
    for player_id in range(0,4):
        print("model id: ", str(model_id), " player id: ", player_id)
        dframe_file = "./results/trained_models/liarsdice/DQN_1K/models-" + str(model_id) + "/" 