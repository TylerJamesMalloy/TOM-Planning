import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os 



all_data = pd.DataFrame()
data = pd.read_pickle('./trained_models/connect_four/MADQN_1M/results.pkl')
np_data = data.to_numpy()
for data_index, datapoint in enumerate(np_data):
    if((data_index - 1) % 3 == 0):
        d = {"Performance": np_data[data_index][0], "Timesteps": np_data[data_index + 1][0], "Loss": np_data[data_index - 1][0]}
        all_data = all_data.append(d, ignore_index=True)


print(all_data)

sns.lineplot(x='Timesteps', y='Performance', data=all_data)
plt.show()

assert(False)
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