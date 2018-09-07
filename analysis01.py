import json
import numpy as np

data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"
# filename = data_path + "results_222236_23072018"
filename = data_path + "results_115159_24072018"


f = open(filename,"r")
experiment_dict = json.load(f)
f.close()

print(experiment_dict['description'])

# print(experiment_dict['trials']['1']['articles'].keys())

for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    score_list = [current_trial[page]['score'] for page in current_trial]
    print(np.mean(score_list), np.std(score_list))
# for trial in experiment_dict['trials']:
#     score_list = [trial[page]['score'] for page in trial]
#     # print(np.mean(score_list),np.std(score_list))



