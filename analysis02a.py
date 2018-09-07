"""
Version of analysis02 that measures precision - i.e the number of LIME or random words that
occur at least once in the annotations (as opposed to counting total number of times
the words appear in the annotations).

"""

import json
import numpy as np
from collections import defaultdict
import glob
import os
from scipy import stats

latest_file = False 

# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"
data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"

# filename = "results_224342_02082018.json"
# filename = "results_110011_09082018.json"
# filename = "results_110011_09082018.json"
filename = "results_095032_17082018.json"
# filename = "results_225909_16082018.json"
# filename = "results_150547_17082018.json"
# file_stem = "results_143523_12082018"
file_stem = "results_214213_18082018"            # current definitive unigram
# file_stem = "results_143523_12082018"            # current definitive unigram

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

for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    debug_count = 0
    raw_exp_counts = []
    raw_rand_counts = []
    article_count = 0 
    for article in current_trial:
        article_count += 1
        # get annontation counts for explainer words
        word_list = current_trial[article]['explainer_words']
        hits =[word_list[word]['annt_count'] for word in word_list]
        exp_precision_count  = 0
        for hit in hits:
            if hit > 0 :
                exp_precision_count += 1
        summary_dict[exp_precision_count]['exp_count'] += 1
        raw_exp_counts.append(exp_precision_count)
        
        # get annontation counts for random words
        random_trials = current_trial[article]['random_words']
        rand_total = 0.0
        rand_count = 0 
        for rt in random_trials[:1]:
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
            raw_rand_counts.append(rand_precision_count)
#         print('article_count, article, rand_count = ',article_count,article, rand_count)
    list_len = max(summary_dict.keys()) + 1 
    
    lime_words_total = np.sum([summary_dict[i]['exp_count'] for i in summary_dict])
    random_words_total = np.sum([summary_dict[i]['random_count'] for i in summary_dict])
    lime_words_percent = np.zeros(list_len,dtype=float)
    random_words_percent = np.zeros(list_len,dtype=float) 
    for i in range(list_len):
        if i in summary_dict:
            lime_words_percent[i] = summary_dict[i]['exp_count'] * 100/ lime_words_total
            random_words_percent[i] = summary_dict[i]['random_count'] * 100 / random_words_total
#             print("%.4d %4d %4d" % (i, summary_dict[i]['exp_count'],summary_dict[i]['random_count']))




# In[2]:

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
  
t_test_and_summary(raw_exp_counts,raw_rand_counts,a_name="LIME words",b_name = "random words")



for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    debug_count = 0
    
    expl_counts = []
    rand_counts = []
    for article in current_trial:
        # get annontation counts for explainer words
        word_list = current_trial[article]['explainer_words']
        hits =[word_list[word]['annt_count'] for word in word_list]
        expl_counts.append(np.sum(hits))
        
        
        # get annontation counts for random words
        random_trials = current_trial[article]['random_words']
        rand_total = 0.0
        for rt in random_trials:
            hits = [rt[word]['annt_count'] for word in rt]
            rand_counts.append(np.sum(hits))



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
ax.set_title('In annotation occurences by word type')
ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
ax.set_xticklabels([str(i) for i in range(n_groups)])

ax.legend()
  
fig.tight_layout()
plt.show()

t_test_and_summary(raw_exp_counts,raw_rand_counts,a_name="LIME words",b_name = "random words") 
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


# In[7]:

print(experiment_dict['trials']['0'].keys())


# In[8]:

print(experiment_dict['trials']['0']['articles']['132'].keys())


# In[9]:

print(experiment_dict['trials']['0']['articles']['132']['explainer_words'].keys())


# In[10]:

print(experiment_dict['trials']['0']['articles']['132']['explainer_words']['obama'].keys())


# In[ ]:



