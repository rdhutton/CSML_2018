"""
Version of analysis02a that measures precision
 - i.e the number of LIME or random words that
occur at least once in the annotations (as opposed to counting total number of times
the words appear in the annotations).

"""

import json
import numpy as np
from collections import defaultdict
import glob
import os
from scipy import stats
import time
##########################################################
#                                                        #
# OPEN AND READ FILE
##########################################################
latest_file = False 

# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"
data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"


# file_stem = "results_214213_18082018"            # current definitive unigram
# file_stem = "results_143523_12082018"            # current definitive bigram (?)
# file_stem = "27082018_170235_S_results"            # first sentence results
file_stem = "results_230453_01092018"            # unigram on v2 of data
# file_stem = "results_131254_06092018_not_bow"    # unigram v2 - not treated as bow
# file_stem = "results_155421_02092018 v2 fair"
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

summary_dict = defaultdict(lambda:{'exp_count':0,'random_count':0})

##########################################################
#                                                        #
# MAKE COUNTS
##########################################################
for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    exp_precision_counts = []
    rand_precision_counts = []
    article_count = 0 
    for article in current_trial:
        article_count += 1
        # get annotation counts for explainer words
        word_list = current_trial[article]['explainer_words']
        hits =[word_list[word]['annt_count'] for word in word_list]
        # count number of explainer words for this article with at least one hit (appearance in annotation)
        exp_precision_count  = 0
        for hit in hits:
            if hit > 0 :
                exp_precision_count += 1
        # increment bin corresponding to this precision count
        summary_dict[exp_precision_count]['exp_count'] += 1
        # add to list of precision counts
        exp_precision_counts.append(exp_precision_count)
        
        # get annontation counts for random words
        random_trials = current_trial[article]['random_words']
        rand_total = 0.0
        rand_count = 0 
        # for rt in random_trials[:1]:
        for rt in random_trials:
            rand_count += 1
            rand_precision_count = 0
            hits = [rt[word]['annt_count'] for word in rt]
            for hit in hits :
                if hit > 0:
                    rand_precision_count += 1
            rand_count1 = sum(x > 0 for x in hits)
            if rand_precision_count != rand_count1:
                print("rand counts don't match !! ")
            summary_dict[rand_precision_count]['random_count'] += 1
            rand_precision_counts.append(rand_precision_count)
    list_len = max(summary_dict.keys()) + 1
    
    lime_words_total = np.sum([summary_dict[i]['exp_count'] for i in summary_dict])
    random_words_total = np.sum([summary_dict[i]['random_count'] for i in summary_dict])
    lime_words_percent = np.zeros(list_len,dtype=float)
    random_words_percent = np.zeros(list_len,dtype=float)
    # convert raw counts to percentages
    for i in range(list_len):
        if i in summary_dict:
            lime_words_percent[i] = summary_dict[i]['exp_count'] * 100/ lime_words_total
            random_words_percent[i] = summary_dict[i]['random_count'] * 100 / random_words_total
            print("%.4d %4d %4d" % (i, summary_dict[i]['exp_count'],summary_dict[i]['random_count']))




##########################################################
#                                                        #
# FUNC for TEST , RUN T TEST
##########################################################

import scipy.stats

def t_test_and_summary(a, b, a_name="sample a", b_name="sample_b", test="t_test"):
    if test == "t_test":
        print("t test \n")
        t_stat, pvalue = scipy.stats.ttest_ind(a, b)
    elif test == "mannwhitney":
        print("Mann Whitney test\n")
        t_stat, pvalue = scipy.stats.mannwhitneyu(a, b)
    else:
        print("Unknown test type : {}".format(t_test))
        return

    print("population sizes = ", len(a), len(b))
    print("%-20s mean = %.2f   sd = %.2f" % (a_name, np.mean(a), np.std(a)))
    print("%-20s mean = %.2f   sd = %.2f" % (b_name, np.mean(b), np.std(b)))
    print()

    print("******** LATEX **********")

    print()

    print("\\begin{tabular}{ |l|c|c| }\n \\hline\n& mean & sd &\n\\hline")
    print("%-20s &  %.2f  & %.2f\\\\" % (a_name, np.mean(a), np.std(a)))
    print("%-20s &  %.2f  & %.2f\\\\" % (b_name, np.mean(b), np.std(b)))
    print("\\hline\\n\\end{tabular}\n\\\\\\\\")

    if test == "t_test":
        print("t statistic = %.4f" % (t_stat), end=' ')
    elif test == "mannwhitney":
        print("Mann Whitney statistic = %.4f" % (t_stat), end=' ')


    if pvalue < 0.001:
        print("  p value$ < 0.001$")
    else:
        print("  p value    $ = %.4f\$" % (pvalue))
  
t_test_and_summary(exp_precision_counts, rand_precision_counts, a_name="LIME words", b_name ="random words")

t_test_and_summary(exp_precision_counts, rand_precision_counts, a_name="LIME words", b_name ="random words",test="mannwhitney")


##########################################################



##########################################################
#                                                        #
# CREATE CHART                                           #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = len(lime_words_percent)


fig, ax = plt.subplots(figsize=(12,6))

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, lime_words_percent, bar_width,
                alpha=opacity, color='b',
                 error_kw=error_config,
                label='LIME words')

rects2 = ax.bar(index + bar_width, random_words_percent, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Random words')

ax.set_xlabel('Occurences in annotation')
ax.set_ylabel('Percent')
# ax.set_title('In annotation occurences by word type')
ax.set_title('Unigram (not bag of words) : annotation occurences')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([str(i) for i in range(n_groups)])

ax.legend()
  
fig.tight_layout()
plt.show()

plots_path = "/Users/rick/Study/CSML/Project/Pycharm/scratch/plots/"
ts = time.localtime()
figfile_name = plots_path + "{}notbow".format(time.strftime("%d%m_%H%M%S", ts))
fig.savefig(figfile_name)

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




