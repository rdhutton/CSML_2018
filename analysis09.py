"""
Version of analysi08.py that compares LIME with positive weighting to LIME with negative weighting

"""

import json
import numpy as np
from collections import defaultdict
import glob
import os
from scipy import stats
##########################################################
#                                                        #
# OPEN AND READ FILE
##########################################################
latest_file = False 

# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"
data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
plot_path = "/Users/rick/Study/CSML/Project/Pycharm/scratch/plots/"

# file_stem = "results_214213_18082018"            # current definitive unigram
# file_stem = "results_143523_12082018"            # current definitive bigram (?)
file_stem = "results_155421_02092018 v2 fair"     # v2 data words tagged 'fair'

filename = file_stem + ".json"

if latest_file == True :
  results_files = data_path + "results*"
  list_of_files = glob.glob(results_files)
  latest_file = max(list_of_files, key=os.path.getctime)
  filepath = latest_file
else:
  filepath = data_path + filename
 
f = open(filepath,"r")
experiment_dict = json.load(f)
f.close()

summary_dict = defaultdict(lambda:{'pos_exp_count':0,'neg_exp_count':0})

##########################################################
#                                                        #
# MAKE COUNTS
##########################################################
for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']

    pos_exp_precision_counts = []
    neg_exp_precision_counts = []

    article_count = 0
    for article in current_trial:
        article_count += 1
        # get annontation counts for explainer words
        word_list = current_trial[article]['explainer_words']
        weights = [word_list[word]['weight'] for word in word_list]
        hits =[word_list[word]['annt_count'] for word in word_list]
        # count number of explainer words for this article with at least one hit (appearance in annotation)
        pos_exp_precision_count  = 0
        neg_exp_precision_count = 0
        # for word in word_list:
        #     print(word, word_list[word]['weight'])
        for i,hit in enumerate(hits):
            if hit > 0 :
                if weights[i] <= 0:
                    pos_exp_precision_count += 1
                else:
                    neg_exp_precision_count += 1
            if hit > 4 :
                print ("{} hits in article{}".format(hit,article))
        # increment bin corresponding to this precision count
        summary_dict[pos_exp_precision_count]['pos_exp_count'] += 1
        summary_dict[neg_exp_precision_count]['neg_exp_count'] += 1
        # add to list of precision counts
        pos_exp_precision_counts.append(pos_exp_precision_count)
        neg_exp_precision_counts.append(neg_exp_precision_count)

#         # get annontation counts for random words
#         random_trials = current_trial[article]['random_words']
#         rand_total = 0.0
#         rand_count = 0
#         # for rt in random_trials[:1]:
#         for rt in random_trials:
#             rand_count += 1
#             rand_precision_count = 0
#             hits = [rt[word]['annt_count'] for word in rt]
#             for hit in hits :
#                 if hit > 0:
#                     rand_precision_count += 1
#             rand_count1 = sum(x > 0 for x in hits)
#             if rand_precision_count != rand_count1:
#                 print("rand counts don't match !! ")
#             summary_dict[rand_precision_count]['random_count'] += 1
#             rand_precision_counts.append(rand_precision_count)
# #         print('article_count, article, rand_count = ',article_count,article, rand_count)
    list_len = max(summary_dict.keys()) + 1 
    
    lime_words_total = np.sum([summary_dict[i]['pos_exp_count'] for i in summary_dict])
    random_words_total = np.sum([summary_dict[i]['neg_exp_count'] for i in summary_dict])
    pos_words_percent = np.zeros(list_len, dtype=float)
    neg_words_percent = np.zeros(list_len, dtype=float)
    # convert raw counts to percentages
    for i in range(list_len):
        if i in summary_dict:
            pos_words_percent[i] = summary_dict[i]['pos_exp_count'] * 100 / lime_words_total
            neg_words_percent[i] = summary_dict[i]['neg_exp_count'] * 100 / random_words_total
            print("%.4d %4d %4d" % (i, summary_dict[i]['pos_exp_count'],summary_dict[i]['neg_exp_count']))




##########################################################
#                                                        #
# FUNC for TEST , RUN T TEST
##########################################################

import scipy.stats

def t_test_and_summary(a,b,a_name="sample a",b_name="sample_b"):
  t_stat,pvalue = scipy.stats.ttest_ind(a,b)
  # t_stat,pvalue = scipy.stats.mannwhitneyu(a,b)
  print("population sizes = ",len(a),len(b))
  print("%-20s mean = %.2f   sd = %.2f"%(a_name,np.mean(a),np.std(a)))
  print("%-20s mean = %.2f   sd = %.2f"%(b_name,np.mean(b),np.std(b)))

  print("%-20s &  %.2f  & %.2f"%(a_name,np.mean(a),np.std(a)))
  print("%-20s &  %.2f  & %.2f"%(b_name,np.mean(b),np.std(b)))

  print("t statistic = %.4f"%(t_stat))
  print("p value     = %.4f"%(pvalue))
  
t_test_and_summary(pos_exp_precision_counts, neg_exp_precision_counts, a_name="LIME words", b_name ="random words")

##########################################################



##########################################################
#                                                        #
# CREATE CHART                                           #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = len(pos_words_percent)


fig, ax = plt.subplots(figsize=(12,6))

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, pos_words_percent, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Pos weight words')

rects2 = ax.bar(index + bar_width, neg_words_percent, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Neg weight words')

ax.set_xlabel('Occurences in annotation')
ax.set_ylabel('Percent')
ax.set_title('Occurence in annotations tagged \"fair\"')
ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
ax.set_xticklabels([str(i) for i in range(n_groups)])

ax.legend()
  
fig.tight_layout()
plt.show()

plt_file = plot_path + "pos_neg_v2_fair"
fig.savefig(plt_file)

                   ##################
      ##############################################
##########################################################



print("\n")
print("notes : ",experiment_dict['description'])
if 'short_desc' in experiment_dict.keys():
  print("notes: ",experiment_dict['short_desc'])


  
print("results file : ",filepath)
print("LIME data cloud size = %d  No of LIME features = %d " %  
      (experiment_dict['trials']['0']['num_samples'],experiment_dict['trials']['0']['num_features']))
if 'rand_sample_size' in experiment_dict['trials']['0'].keys():
        print("random sample size = ",experiment_dict['trials']['0']['rand_sample_size'])
if 'stop_words' in experiment_dict['trials']['0'].keys():
        print("stop_words = ",experiment_dict['trials']['0']['stop_words'])




