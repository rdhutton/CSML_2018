"""

For analyzing sentence data

Based on analysis08.

measures precision - i.e the number of LIME or random words that
occur at least once in the annotations (as opposed to counting total number of times
the words appear in the annotations).

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

# file_stem = "results_214213_18082018"            # current definitive unigram
# file_stem = "results_143523_12082018"            # current definitive bigram (?)
file_stem = "27082018_170235_S_results"  # first sentence results
filename = file_stem + ".json"

if latest_file == True:
    results_files = data_path + "results*"
    list_of_files = glob.glob(results_files)
    latest_file = max(list_of_files, key=os.path.getctime)
    filepath = latest_file
else:
    filepath = data_path + filename

f = open(filepath, "r")
experiment_dict = json.load(f)
f.close()

summary_dict = defaultdict(lambda: {'exp_count': 0, 'random_count': 0})

##########################################################
#                                                        #
# MAKE COUNTS
##########################################################
for trial in experiment_dict['trials']:
    current_trial = experiment_dict['trials'][trial]['articles']
    exp_precision_scores = []
    rand_precision_scores = []
    article_count = 0
    for article in current_trial:
        article_count += 1
        # get annontation counts for explainer words
        word_list = current_trial[article]['explainer_words']
        hits = [word_list[word]['annt_count'] for word in word_list]
        exp_precision_scores.extend(hits)
        # count number of explainer words for this article with at least one hit (appearance in annotation)

        # get annontation counts for random words
        random_trials = current_trial[article]['random_words']
        rand_total = 0.0
        rand_count = 0
        # for rt in random_trials[:1]:
        for rt in random_trials:
            rand_count += 1
            rand_precision_count = 0
            hits = [rt[word]['annt_count'] for word in rt]
            rand_precision_scores.extend(hits)
    hist,bin_edges = np.histogram(exp_precision_scores,bins=10)
    print(hist)
    print(bin_edges)
    r_hist,r_bin_edges = np.histogram(rand_precision_scores,bins=10)
    print(r_hist)
    print(r_bin_edges)


##########################################################
#                                                        #
# FUNC for TEST , RUN T TEST
##########################################################

import scipy.stats


def t_test_and_summary(a, b, a_name="sample a", b_name="sample_b"):
    t_stat, pvalue = scipy.stats.ttest_ind(a, b)
    # t_stat,pvalue = scipy.stats.mannwhitneyu(a,b)
    print("population sizes = ", len(a), len(b))
    print("%-20s mean = %.4f   sd = %.4f" % (a_name, np.mean(a), np.std(a)))
    print("%-20s mean = %.4f   sd = %.4f" % (b_name, np.mean(b), np.std(b)))
    print()

    print("******** LATEX **********")

    print()

    print("\\begin{tabular}{ |l|c|c| }\n \\hline\n& mean & sd &\n\\hline")
    print("%-20s &  %.2f  & %.2f\\\\" % (a_name, np.mean(a), np.std(a)))
    print("%-20s &  %.2f  & %.2f\\\\" % (b_name, np.mean(b), np.std(b)))
    print("\\hline\\n\\end{tabular}\n\\\\\\\\")

    print("t statistic = %.4f" % (t_stat), end=' ')
    if pvalue < 0.001:
        print("  p value$ < 0.001$")
    else:
        print("  p value    $ = %.4f\$" % (pvalue))


t_test_and_summary(exp_precision_scores, rand_precision_scores, a_name="LIME sentences", b_name="Random sentences")

##########################################################

bin_num = 10
lime_words_total = len(exp_precision_scores)
random_words_total = len(rand_precision_scores)
lime_words_percent = np.zeros(bin_num, dtype=float)
random_words_percent = np.zeros(bin_num, dtype=float)
for i in range(bin_num):
    lime_words_percent[i] = hist[i] / lime_words_total
    random_words_percent[i] = r_hist[i] / random_words_total
##########################################################
#                                                        #
# CREATE CHART                                           #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = len(lime_words_percent)

fig, ax = plt.subplots(figsize=(12, 6))

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, lime_words_percent, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='LIME sentences')

rects2 = ax.bar(index + bar_width, random_words_percent, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Random sentences')

ax.set_xlabel('Proportion overlapping with annotation')
ax.set_ylabel('Percent')
ax.set_title('Annotation overlap by sentence type')
ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
ax.set_xticklabels([str(x/10)+"-"+str((x+1)/10) for x in range(10)])

ax.legend()

fig.tight_layout()
plt.show()

##################
##############################################
##########################################################


print("\n")
print("notes : ", experiment_dict['description'])
if 'short_desc' in experiment_dict.keys():
    print("notes: ", experiment_dict['short_desc'])

print("results file : ", filepath)
print("LIME data cloud size = %d  No of LIME features = %d " %
      (experiment_dict['trials']['0']['num_samples'], experiment_dict['trials']['0']['num_features']))
if 'rand_sample_size' in experiment_dict['trials']['0'].keys():
    print("random sample size = ", experiment_dict['trials']['0']['rand_sample_size'])
if 'stop_words' in experiment_dict['trials']['0'].keys():
    print("stop_words = ", experiment_dict['trials']['0']['stop_words'])

    def as_pyplot_figure(self, label=1, **kwargs):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        return fig




