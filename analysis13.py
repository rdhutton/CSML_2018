"""
per size of data cloud 500,1000...5000
show average score (R^2)
"""

import json
import pandas as pd
import numpy as np
import textwrap
from collections import defaultdict
import glob
import os
from collections import defaultdict

data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"

# experiment_dict = {'timestamp':timestamp,
#                    'description': desc_string ,
#                    'short_desc' : short_desc,
#                      'trials':
#                        {str(trial):
#                          {
#                            'num_articles':num_articles,
#                            'num_features':num_features,
#                            'num_samples':num_samples,
#                            'rand_sample_size':rand_control_num,
#                            'stop_words': stop_words,
#                            'corpus':test_file,
#                            'run_time':'',
#                            'articles':
#                              {str(page_id):
#                                {
#                                'score':0.0,
#                                'txt_len':0,
#                                'quote_len':0,
#                                'explainer_words':[],
#                                'random_words':[]
#                                } for page_id in page_id_list[:num_articles]
#                              }
#                          }for trial in range(num_trials)
#                        }
#                   }


filepath = data_path + "results_grid_25082018_224941.json"


f = open(filepath, "r")
try:
    experiment_dict = json.load(f)
except:
    f.close()
else:
    for trial_no in range(1,6):
        trial_key = str(trial_no)
        # print("notes : ", experiment_dict['description'])
        # if 'short_desc' in experiment_dict.keys():
        #     print("notes: ", experiment_dict['short_desc'])
        # print("rand_sample_size",experiment_dict['trials'][trial_key]['rand_sample_size'])
        # if 'stop_words' in experiment_dict['trials'][trial_key].keys():
        #     print("stop_words = ", experiment_dict['trials'][trial_key]['stop_words'])
        f.close()
        control = 0
        print("trial no = ",trial_no)
        score_list = [ experiment_dict['trials'][trial_key]['articles'][article]['score'] for article in \
                      experiment_dict['trials'][trial_key]['articles']]
        print(len(score_list),np.mean(score_list),np.std(score_list))