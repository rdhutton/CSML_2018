"""

Shows a selected word in context (first found occurrence in a text)


"""
import pandas as pd
import textwrap
import re

test_file = "/Users/rick/factmata/train.csv"
testing = pd.read_csv(test_file)

def wrap_print(text, line_len):
    wrapped_text = ''
    text_lines = textwrap.wrap(text, width=line_len, replace_whitespace=False)
    for line in text_lines:
        wrapped_text = wrapped_text + line + "\n"
    print(wrapped_text)


# for text in testing['text']:
#     loc = text.lower().find("foster ")
#     if(loc != -1):
#         wrap_print(text[max(0,loc-200):min(len(text),loc+200)],90)
#         print("\n")

target = "oq3c4zwd2u"
for i in range(len(testing)):
    text = testing.loc[i]['text']x
    loc = text.lower().find(target)
    if(loc != -1):
        print(i,"," ,testing.loc[i]['tag'],",")
        wrap_print(text[max(0,loc-200):min(len(text),loc+200)],90)
        print("\n")

        regex = re.compile(target, re.IGNORECASE)
        occurences = sum(1 for i in regex.finditer(text))
        print("occurences", occurences)
        print("length of text = ", len(text))


