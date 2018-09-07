"""
Version of analysis that uses precision
I.e. for LIME and random words - counts the number which are found in the annotation
(just counting once however many times they appear)

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


# ## Text colour utilities


RED_FOREGROUND = "\x1b[31m"
BLUE_BACKGROUND = "\x1b[44m"
NORMAL = "\x1b[0m"


# def colour_words(text, target_word, colour_esc):
#     location = -1
#     # Loop while true.
#     locations = []
#     target_len = len(target_word)
#     while True:
#         location = text.find(target_word, location + 1)
#
#         # Break if not found.
#         if location == -1:
#             break
#         else:
#             locations.append((location, location + target_len))
#     print("type of location = ", type(locations))
#
#     locations.reverse()
#     for loc in locations:
#         (start, stop) = loc
#         text = text[:stop] + NORMAL + text[stop:]
#         text = text[:start] + colour_esc + text[start:]
#
#     # Display result.
#     # print()
#     # print(text)
#     return (text)


# def colour_regions(text, regions, colour_esc):
#     regions.sort()
#     regions.reverse()
#     for region in regions:
#         (start, stop) = region
#         text = text[:stop] + NORMAL + text[stop:]
#         text = text[:start] + colour_esc + text[start:]
#
#     # Display result.
#     # print()
#     # print(text)
#     return (text)


##  function to return all rows (i.e. annotations) matching a page_id
# def get_quotes_list(testing, page_id):
#     """
#      Args:
#        testing: panda data frame with text, quotes etc
#        page_id: id of piece of text (as represented in 'page__id' of data frame)
#
#      Returns:
#        List of all rows in data frame that match page_id (i.e. all rows which represent
#        quotes from this piece of text)
#     """
#
#     #   TODO improve error handling ?
#     #   if page_id not in testing['page__id']:
#     #     return []
#     page_refs = (testing['page__id'] == page_id)
#     pages_list = testing[page_refs]
#     return pages_list


import random


# def build_quoted_regions(quoted_parts, randomize=False):
#     """
#     sort quoted parts, then merge any which overlap or form
#     continuous quotes
#
#     Args :
#         list of tuples representing sections of text
#     Returns:
#         The same but with sections merged to continuous regions where
#         appropriate
#     """
#
#     def purge_list(target_list, remove_list):
#         for i in reversed(remove_list):
#             del (target_list[i])
#         return target_list
#
#     remove_list = []
#     ####  remove any dodgy tuples ####
#     # check for tuples with start point after stop point, or negative start
#     for i in range(len(quoted_parts)):
#         start, stop = quoted_parts[i]
#         if (start < 0) or (start > stop):
#             print("warning - start point should be non-negative and not greater than stop point in ", quoted_parts[i])
#             remove_list.append(i)
#     # for i in reversed(remove_list):
#     #     del(quoted_parts[i])
#     purge_list(quoted_parts, remove_list)
#
#     #     print(quoted_parts) # debug
#     #### main sort and merge ####
#     quoted_parts = sorted(quoted_parts)
#     quoted_parts_len = len(quoted_parts)
#     remove_list = []
#     for i in range(quoted_parts_len - 1):
#         # compare end of one part with start of other - merge if overlap
#         start1, stop1 = quoted_parts[i]
#         start2, stop2 = quoted_parts[i + 1]
#         if stop1 >= start2:
#             if stop2 <= stop1:
#                 # remove subsumed part
#                 remove_list.append(i + 1)
#             else:
#                 # second part absorbs first, first is removed
#                 quoted_parts[i + 1] = (start1, stop2)
#                 remove_list.append(i)
#     purge_list(quoted_parts, remove_list)
#
#     return quoted_parts


##############################

# def get_quote_idxs(txt, quotes):
#     """
#     Args:
#         txt - string
#         quotes - list of string
#     Returns:
#         list of tuples indicating (start,stop) position of each quote in txt (if doesn't appear it's ignored)
#     """
#     quoted_idxs = []
#     for quote in quotes:
#         start = txt.find(quote)
#         if start != -1:
#             quoted_idxs.append((start, start + len(quote)))
#
#     return quoted_idxs


# *************************************************** #
# *************************************************** #
#           START OF SCRIPT
# *************************************************** #
# *************************************************** #

latest_file = False

# data_path = "/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/"

data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
plots_path = "/Users/rick/Study/CSML/Project/Pycharm/scratch/plots/"

# file_stem = "results_230453_01092018"              # unigram , new data set
# file_stem = "results_214213_18082018"            # current definitive unigram - first data set
file_stem = "results_143523_12082018"            # current definitive bigram (?)
# file_stem = "27082018_170235_S_results"            # first sentence results
filename = file_stem + ".json"

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

print("file = ", filepath)
f = open(filepath, "r")
experiment_dict = json.load(f)
f.close()

# test_file = "/Users/rick/factmata/article_quotes.csv"
# test_file = "/Users/rick/factmata/article quotes v2.csv"
test_file = "/Users/rick/factmata/article quotes - 2018-08-09.csv"
testing = pd.read_csv(test_file)

f = open(filepath, "r")
experiment_dict = json.load(f)
f.close()
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
weights_dict = defaultdict(list)
for text_id in texts:
    # print(articles[132])
    for word in texts[text_id]['explainer_words']:
        weights_dict[word].append(texts[text_id]['explainer_words'][word]['weight'])
    importance_values = []
for item in weights_dict:
    importance = (sum([abs(value) for value in (weights_dict[item])])**0.5 )
    # print("%s  %.3f"%(item, importance))
    importance_values.append((item, importance))
    importance_dict = dict(importance_values)
print(list(reversed(sorted(importance_values))))
importance_list = list(reversed(sorted(importance_values)))
sorted_by_importance = list(reversed(sorted(importance_values, key=lambda tup: tup[1])))
# print("Top 10")
# for i,(word,weight) in enumerate(sorted_by_importance[:10]):
#     print(i,"%-35s  \t\t%.3f"%(word,weight))
# print("Bottom 10")
# for i,(word,weight) in enumerate(sorted_by_importance[-10:]):
#     print(i,"%-35s  \t\t%.3f"%(word,weight))
print("Top 10")
for i,(word,weight) in enumerate(sorted_by_importance):
    print(i,"%-35s  \t\t%.3f"%(word,weight))

def as_pyplot_figure(exp, class_label,x_min,x_max,art_title):
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
    # explanationsp = self.as_list(label=label, **kwargs)
    fig = plt.figure()
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ['green' if x > 0 else 'red' for x in vals]
    pos = np.arange(len(exp)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    plt.xlim([x_min,x_max])
    # title = 'Local explanation for class %s' % class_label
    title = art_title
    plt.title(title)
    plt.show()
    return fig

greedy_list = []
greedy_titles = []
for _ in range(25):
    max_importance = 0
    max_text_id = -1
    for text_id in texts:
        sum_of_importance = sum([importance_dict[word] for word in texts[text_id]['explainer_words']])
        if sum_of_importance > max_importance:
            max_importance = sum_of_importance
            max_text_id = text_id
            # once selected a word will count zero in any new sums
    greedy_list.append(max_text_id)
    greedy_titles.append(get_title(testing, max_text_id))
    for word in texts[max_text_id]['explainer_words']:
        importance_dict[word] = 0

# first pass to get max, min values - used in drawing plots
wts_list = []
for j,text_id in enumerate(greedy_list):
    for i,word in enumerate(texts[text_id]['explainer_words']):
        wts_list.append(-texts[text_id]['explainer_words'][word]['weight'])
max_wt = max(wts_list) + 0.02
min_wt = min(wts_list) - 0.02

ts = time.localtime()

results_file = "results/results_" + time.strftime("%H%M%S_%d%m%Y.json", ts)

new_importance_dict = dict(importance_list)
print("lens",len(greedy_list),len(greedy_titles))
for j,text_id in enumerate(greedy_list):
    list_sum = 0
    exp_list = []
    for i,word in enumerate(texts[text_id]['explainer_words']):
        print("{} {:20}{:.3f}".format(text_id, word,new_importance_dict[word]),end= ' ')

        # print("{} {:20}{:.3f}".format(text_id, word,texts[text_id]['explainer_words'][word]['weight']),end= ' ')
        if i == 4:
            print("")

        list_sum += new_importance_dict[word]
        new_importance_dict[word] = 0
        # exp_list.append((word,new_importance_dict[word]))

        # print("weight = ", weights_dict[word])

        # exp_list.append((word,weights_dict[word]))
        # print('\n')
        # print(">>>> ",word,texts[text_id]['explainer_words'][word]['weight'])
        # LIME has positive and negative opposite ways around - so negate the weight
        exp_list.append((word, -texts[text_id]['explainer_words'][word]['weight']))
        new_importance_dict[word] = 0
    print("\ntotal value = ",list_sum)
    list_sum = 0


    # retrieve original weights

    exp_list = sorted(exp_list,key = lambda x:x[1],reverse=True)
    wrap_title = "\n".join(wrap(greedy_titles[j], 60))
    print("EXP LIST ")
    print(exp_list)
    fig = as_pyplot_figure(exp_list,"hyperpartisan ",min_wt,max_wt, wrap_title)
    figfile_name = plots_path + "{}bigram_splime{serial:02d}".format(time.strftime("%d%m_%H%M%S", ts),serial=j)

    fig.savefig(figfile_name)





