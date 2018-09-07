import glob
import json
import os
import re
import textwrap

import pandas as pd

# ## Text colour utilities

"""
print("\033[34;42mMy text\033[m")
will print My text in blue on a background of green.

The escape sequence is \033[ followed by ;-separated numbers, followed by m. To end the color, you use \033[m. 
The numbers are 1 to make the text bold, 3 with another number to make the text the color of the other number, 
and 4 with another number to make the text background the color of the other number. 
The other numbers mentioned for 3 and 4 are the following:

0 -> Black
1 -> Red
2 -> Green
3 -> Yellow
4 -> Blue
5 -> Purple
6 -> Cyan
7 -> Light Gray
"""

RED_FOREGROUND = "\x1b[31m"
BLUE_BACKGROUND = "\x1b[44m"
RED_BLUE = "\x1b[31;44m"
NORMAL = "\x1b[0m"


def colour_words(text, target_word, colour_esc):
    location = -1
    # Loop while true.
    locations = []
    target_len = len(target_word)
    while True:
        location = text.find(target_word, location + 1)

        # Break if not found.
        if location == -1:
            break
        else:
            locations.append((location, location + target_len))
    print("type of location = ", type(locations))

    locations.reverse()
    for loc in locations:
        (start, stop) = loc
        text = text[:stop] + NORMAL + text[stop:]
        text = text[:start] + colour_esc + text[start:]

    # Display result.
    # print()
    # print(text)
    return (text)


def colour_regions(text, regions, colour_esc):
    regions.sort()
    regions.reverse()
    for region in regions:
        (start, stop) = region
        text = text[:stop] + NORMAL + text[stop:]
        text = text[:start] + colour_esc + text[start:]

    return (text)

def find_overlay(region, existing_regions):
    """
    TO DO - add case where region straddles two or more existing regions

    :param region: new substring
    :param existing_regions: existing substrings
    :return: (False,_,_) if new substring doesn't overlap
            (True, start,stop) if it does
    """
    overlap_start = 0
    overlap_stop = 0
    existing_start = 0
    existing_stop = 0
    overlaid = False

    new_start, new_stop = region
    for (start,stop) in existing_regions:
        if overlaid :
            break
        if start <= new_start < stop:
            overlaid = True
            existing_start = start
            existing_stop = stop
            overlap_start = new_start
            if new_stop <= stop:
                overlap_stop = new_stop
            else:
                overlap_stop = stop

    return overlaid,overlap_start,overlap_stop,existing_start,existing_stop




def overlay_colour_regions(text, new_regions, coloured_regions, new_colour, old_colour, mixed_colour):
    new_regions.sort()
    new_regions.reverse()
    for region in new_regions:
        (start, stop) = region
        overlaid,overlay_start,overlay_end,ex_start,ex_end = find_overlay(region,coloured_regions)
        if not overlaid:
            text = text[:stop] + NORMAL + text[stop:]
            text = text[:start] + new_colour + text[start:]
        else:
            tmp_buf = text[:ex_start] + old_colour + text[ex_start:overlay_start] + mixed_colour + \
                text[overlay_start:overlay_end] + old_colour + text[overlay_end:ex_end] + NORMAL + text[ex_end:]
            text = tmp_buf

    return (text)
##  function to return all rows (i.e. annotations) matching a page_id
def get_quotes_list(testing, page_id):
    """
   Args:
     testing: panda data frame with text, quotes etc
     page_id: id of piece of text (as represented in 'page__id' of data frame)
   
   Returns:
     List of all rows in data frame that match page_id (i.e. all rows which represent
     quotes from this piece of text)
  """

    #   TODO improve error handling ?
    #   if page_id not in testing['page__id']:
    #     return []
    page_refs = (testing['page__id'] == page_id)
    pages_list = testing[page_refs]
    return pages_list


def build_quoted_regions(quoted_parts, randomize=False):
    """
    sort quoted parts, then merge any which overlap or form
    continuous quotes

    Args :
        list of tuples representing sections of text
    Returns:
        The same but with sections merged to continuous regions where
        appropriate
    """

    def purge_list(target_list, remove_list):
        for i in reversed(remove_list):
            del (target_list[i])
        return target_list

    remove_list = []
    ####  remove any dodgy tuples ####
    # check for tuples with start point after stop point, or negative start
    for i in range(len(quoted_parts)):
        start, stop = quoted_parts[i]
        if (start < 0) or (start > stop):
            print("warning - start point should be non-negative and not greater than stop point in ", quoted_parts[i])
            remove_list.append(i)
    # for i in reversed(remove_list):
    #     del(quoted_parts[i])
    purge_list(quoted_parts, remove_list)

    #     print(quoted_parts) # debug
    #### main sort and merge ####
    quoted_parts = sorted(quoted_parts)
    quoted_parts_len = len(quoted_parts)
    remove_list = []
    for i in range(quoted_parts_len - 1):
        # compare end of one part with start of other - merge if overlap
        start1, stop1 = quoted_parts[i]
        start2, stop2 = quoted_parts[i + 1]
        if stop1 >= start2:
            if stop2 <= stop1:
                # remove subsumed part
                remove_list.append(i + 1)
            else:
                # second part absorbs first, first is removed
                quoted_parts[i + 1] = (start1, stop2)
                remove_list.append(i)
    purge_list(quoted_parts, remove_list)

    return quoted_parts


##############################

def get_quote_idxs(txt, quotes):
    """
    Args:
        txt - string
        quotes - list of string
    Returns:
        list of tuples indicating (start,stop) position of each quote in txt (if doesn't appear it's ignored)
    """
    quoted_idxs = []
    for quote in quotes:
        start = txt.find(quote)
        if start != -1:
            quoted_idxs.append((start, start + len(quote)))

    return quoted_idxs


# *************************************************** #
# *************************************************** #
#           START OF SCRIPT
# *************************************************** #
# *************************************************** #


data_path = "/Users/rick/factmata/factnlp-experimental/explainability/results/"
# filename = "results_163027_27072018"

# list_of_files = glob.glob('/Users/rick/factmata/factnlp-experimental/hyperpartisanship/datasets/results*')
list_of_files = glob.glob('/Users/rick/factmata/factnlp-experimental/explainability/results/results*')
latest_file = max(list_of_files, key=os.path.getctime)

filename = "results_095032_17082018.json"
filepath = data_path + filename

# filepath = latest_file

test_file = "/Users/rick/factmata/article_quotes.csv"
testing = pd.read_csv(test_file)

print("file = ", filepath)
f = open(filepath, "r")
experiment_dict = json.load(f)
f.close()

articles = [article_id for article_id in experiment_dict['trials']['0']['articles']]
# select article
# PAGE_ID = 32
for PAGE_ID in range(len(articles)):
# for PAGE_ID in [16,54,80,87,101,102,105,114,150]:
    page_id = int(articles[PAGE_ID])

    print("*** LIME words ***")
    current_trial = experiment_dict['trials']['0']
    word_list = current_trial['articles'][str(page_id)]['explainer_words']
    hits = [word_list[word]['annt_count'] for word in word_list]
    # for key in current_trial['articles'][str(page_id)]['explainer_words'].keys():
    #
    #     print(key,word_list[key]['annt_count'])
    word_strings = []
    for word in word_list:
        # print(word, word_list[word]['annt_count'], word_list[word]['text_count'], end = ' * ')
        print(word,  end = ' ,')
        word_strings.append(word)
    print("\n")

    text = testing[testing['page__id'] == page_id].iloc[0]['text']
    text = ' '.join(text.splitlines())
    wrapped_text =''
    text_lines = textwrap.wrap(text, width=120, replace_whitespace=False)
    for line in text_lines:
        wrapped_text = wrapped_text + line + "\n"

    # Get text of article and find all quotes for given article
    quotes_list_df = get_quotes_list(testing, page_id)
    quotes_list = [quote for quote in quotes_list_df['quote']]
    quotes_idxs = get_quote_idxs(text, quotes_list)
    quote_regions = build_quoted_regions(quotes_idxs)

    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # wrapped_text = colour_regions(wrapped_text, quote_regions, RED_FOREGROUND)

    ############################################
    # find location of LIME words and add colour
    match_locs = []
    for word in word_strings:
        p = re.compile(r"\W%s\W" % (word))
        # s = p.search(text)
        for match in p.finditer(wrapped_text.lower()):
            match_locs.append((match.start() + 1, match.end() - 1))
    wrapped_text = colour_regions(wrapped_text, match_locs, BLUE_BACKGROUND)
    # wrapped_text = overlay_colour_regions(wrapped_text,match_locs,quote_regions,BLUE_BACKGROUND,RED_FOREGROUND,RED_BLUE)
    #                                          #
    ############################################


    ############################################
    # print final version                      #
    print("article id = ", PAGE_ID,'/',page_id)

    # wrapped_text = textwrap.wrap(text, width=120, replace_whitespace=False)
    # for line in wrapped_text:
    #     print(line.lower())

    print(wrapped_text)
    #                                          #
    ############################################


    # print("*** Random words ***")
    # random_trials = current_trial['articles'][str(page_id)]['random_words']
    # for rt in random_trials[:2]:
    #     for key in rt.keys():
    #         print(key, rt[key]['annt_count'], rt[key]['text_count'])
