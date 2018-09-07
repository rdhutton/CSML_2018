import pandas as pd
import re
import numpy as np

test_file = "/Users/rick/factmata/article_quotes.csv"
# test_file = "/Users/rick/factmata/article quotes - 2018-08-09.csv"
testing = pd.read_csv(test_file)

page_id_set = set(testing['page__id'])
word_counts = []
for article in page_id_set :
    text = testing[testing['page__id'] == article].iloc[0]['text']
    word_counts.append(len(re.split(r'\W+',  text)))
word_counts = list(filter(lambda x: x > 100,word_counts))
print("Number of words per article : mean = %.1f  sd = %.2f  min = %d   max = %d" % (np.mean(word_counts),np.std(word_counts),min(word_counts),max(word_counts)))
print ("Total number of quotes", len(word_counts))
print()
# number of quotes per article
# for article in page_id_set:
#     for item in testing[testing['page__id'] == article]:
#
#         print(article,item['quote'])

quote_counts = []
page_id_set_list = list(page_id_set)
for page in page_id_set_list:
    for item in testing[testing['page__id'] == page]['quote']:
        quote_counts.append(len(re.split(r'\W+',  item)))
print("Total number of quotes = %d   Mean quotes per article = %.2f" % (len(quote_counts),len(quote_counts)/len(page_id_set_list)))
print("Number of words per quote : mean = %.1f  sd = %.2f  min = %d   max = %d"%(np.mean(quote_counts), np.std(quote_counts),min(quote_counts),max(quote_counts)))





