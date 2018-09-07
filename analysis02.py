import json
import numpy as np
from collections import defaultdict

data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"
# filename = data_path + "results_222236_23072018"
filename = data_path + "results_143523_12082018" +".json"

f = open(filename,"r")
experiment_dict = json.load(f)
f.close()

summary_dict = defaultdict(lambda:{'exp_count':0,'random_count':0})

exp_grand_total = 0
rand_grand_total = 0.0
for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    debug_count = 0
    for article in current_trial:
        # get annontation counts for explainer words
        # print("score = ", current_trial[article]['score'])
        word_list = current_trial[article]['explainer_words']
        hits =[word_list[word]['annt_count'] for word in word_list]
        exp_total = np.sum(hits)
        summary_dict[exp_total]['exp_count'] += 1
        # get annontation counts for random words
        random_trials = current_trial[article]['random_words']
        rand_total = 0.0
        for rt in random_trials:
            hits = [rt[word]['annt_count'] for word in rt]
            print(hits)
            # for word in rt:
            #     hit=rt[word]['annt_count']
            #     print(word,hit)
            rand_total = np.sum(hits)
            summary_dict[rand_total]['random_count'] += 1
            # summary_dict[exp_total]['randoms'].append(rand_total)
        rand_total = rand_total / len(rt)



    for i in range(max(summary_dict.keys())):
        if i in summary_dict.keys():
            print("%.4d %4d %4d" % (i, summary_dict[i]['exp_count'],summary_dict[i]['random_count']))

        # print("grand totals = ", exp_grand_total,rand_grand_total)

