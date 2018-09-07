"""
compare prob returned by the model
with score on linear reg. by LIME

"""


import json
import pandas as pd
import numpy as np
import textwrap
from collections import defaultdict
import glob
import os
from collections import defaultdict
from textwrap import wrap
import time
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr



##############################



# *************************************************** #
# *************************************************** #
#           START OF SCRIPT
# *************************************************** #
# *************************************************** #

latest_file = False

# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"

data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
plots_path = "/Users/rick/Study/CSML/Project/Pycharm/scratch/plots/"

file_stem = "results_230453_01092018"              # unigram , new data set
# file_stem = "results_214213_18082018"            # current definitive unigram - first data set
# file_stem = "results_143523_12082018"            # current definitive bigram (?)
# file_stem = "27082018_170235_S_results"            # first sentence results
probs_file_stem = "results_082755_04092018 probs"            # file of probabilites from underlying model
filename = file_stem + ".json"
probs_file = probs_file_stem +".json"

# data_path = "/Users/rick/factmata/factnlp-experimental/explainability/"
#
# # filename = "results_224342_02082018.json"
# # filename = "results_143523_12082018.json"
# filename = "results_220909_17082018.json"

if latest_file == True:
    results_files = data_path + "results*"
    list_of_files = glob.glob(results_files)
    latest_file = max(list_of_files, key=os.path.getctime)
    filepath = latest_file
else:
    filepath = data_path + filename

probs_filepath = data_path + probs_file

print("file = ", filepath)

f = open(filepath, "r")
experiment_dict = json.load(f)
f.close()

pf = open(probs_filepath, "r")
probs_dict = json.load(pf)
pf.close()
# test_file = "/Users/rick/factmata/article_quotes.csv"
# test_file = "/Users/rick/factmata/article quotes v2.csv"
test_file = "/Users/rick/factmata/article quotes - 2018-08-09.csv"
testing = pd.read_csv(test_file)


def get_title(testing, page_id):

    """
    :param page_id: identifies an article in the
    article_pd :
    :return: string containing title of article
    """
    try:
        return(testing[testing['page__id'] == int(page_id)].iloc[0]['title'])
    except:
        print("failed to find title , page_id = {}".format(page_id))

articles = [article_id for article_id in experiment_dict['trials']['0']['articles']]

# select article
# page_id = int(articles[30])

texts = experiment_dict['trials']['0']['articles']

probs = []
LR_scores = []
LR_lead_wts = []
LR_all_wts = []
for article in texts:
    if article in experiment_dict['trials']['0']['articles']:
        probs.append(probs_dict['trials']['0']['articles'][article]['model_prob'])
        LR_scores.append(experiment_dict['trials']['0']['articles'][article]['score'])
        weights_list = list([texts[article]['explainer_words'][word]['weight'] for word in \
                        texts[article]['explainer_words']])
        LR_lead_wts.append(np.mean(weights_list[:1]))
        LR_all_wts.append(np.mean(weights_list))

    else:
        print("Probs file : No key found for {}".format(article))

print("mean = {:.3f}  sd = {:.3f}  n = {}".format(np.mean(probs),np.std(probs),len(probs)))
print("mean = {:.3f}  sd = {:.3f}  n = {}".format(np.mean(LR_scores),np.std(LR_scores),len(LR_scores)))

plt.scatter(LR_scores,probs)
plt.show()

coeff, p = pearsonr(LR_scores, probs)
print("prob/score   Correlation coefficient = {:.3f}  p value = {:.3f}".format(coeff,p))
coeff, p = pearsonr(LR_lead_wts, probs)
print("prob/ld wts  Correlation coefficient = {:.3f}  p value = {:.3f}".format(coeff,p))
coeff, p = pearsonr(LR_all_wts, probs)
print("prob/all wts Correlation coefficient = {:.3f}  p value = {:.3f}".format(coeff,p))
