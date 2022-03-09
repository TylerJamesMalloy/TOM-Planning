import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os 

MODEL = "MADQN_10K"
ENV = "tiktaktoe"

dict = {'Model':[],
        'Opponent':[],
        'Timesteps':[],
        'Reward':[]
       }
  
all_data = pd.DataFrame(dict)
for model_id in range(0,101):
    for player_id in range(0,2):
        for opponent in ['random', 'models-100']:
            if(opponent == 'random'):
                numpy_path = "./trained_models/" + ENV + "/" + MODEL + "/models-" + str(model_id) + "/p" + str(player_id) + "/random/results.npy" 
            else:
                numpy_path = "./trained_models/" + ENV + "/" + MODEL + "/models-" + str(model_id) + "/p" + str(player_id) + "/models-100/results.npy" 
            
            if(not os.path.exists(numpy_path)): continue 

            data = np.load(numpy_path)
            player_data = data[:,player_id]

            for index, reward in enumerate(player_data):
                #if(index > 10): continue 
                all_data.loc[len(all_data.index)] = [MODEL, opponent, model_id, reward] 

sns.lineplot(x='Timesteps', y='Reward', hue='Opponent', data=all_data)
plt.show()
print(all_data)