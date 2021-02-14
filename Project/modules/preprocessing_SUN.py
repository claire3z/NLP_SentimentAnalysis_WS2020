
''"""
Created by: Claire Z. Sun
Date: 2021.01.13
''"""

### Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast    # bytes to string
import re
import emoji
import spacy # stopwords
from nltk.stem import PorterStemmer
from symspellpy import SymSpell, Verbosity # spelling correction
import pkg_resources


### Set-up Preparation

# Set dataframe display options - displaying all contents in all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# Prepare initial spell checker and load necessary resources
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# Prepare emoticon and slang dictionaries
dict_emoticon = pd.read_csv('./modules/emoticons.csv', header='infer', delimiter='\t', index_col=0).T.to_dict()
dict_acronym = pd.read_csv('./modules/acronym.csv', header='infer', delimiter=',', index_col=0).T.to_dict()

# Prepare for tokenization and load stopwords
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
stopwords = nlp.Defaults.stop_words

# Initialize the stemmer from NLTK
stemmer = PorterStemmer()


### Preprocessing pipeline for Twitter texts
#emoji_counter = 0
def tweet_preprocessing(text,
                        _url=True, _handle=True, _hashtag=True, _elongation=True, _negation=True, _emoji=True,  # stage-1
                        _emoticon=True, _acronym=True,  # stage-2
                        _spelling=False, _numeric=False, _specialChar=True):  # stage-3
    #global emoji_counter

    ### Stage 1: regular expression-based processes performing on whole string

    # remove urls
    if _url:
        text = re.compile(r'(http\S+)').sub(r'', text)

    # remove @usernames
    if _handle:
        # text = re.compile(r'(@\w+)').sub(r'', text) # remove both @ and username
        text = re.compile(r'(@\w+)').sub(r'USERNAME', text)  # replace @handle with generic USERNAME

    # remove # in #hashtags but keep the text in tags

    if _hashtag:
        # text = re.compile(r'(#\w+)').sub(r'', text)
        text = re.compile(r'(#)').sub(r'', text)

    # dealing with negation [before removing special chars]
    if _negation:
        text = re.compile(r"won't").sub(r'will not', text)
        text = re.compile(r"can't").sub(r'can not', text)
        text = re.compile(r"ain't").sub(r'am not', text)
        text = re.compile(r"n't").sub(r' not', text)

    # restoring elongated words to original form - RE
    # Ref: https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
    if _elongation:
        text = re.compile(r"(.)\1{2,}").sub(r"\1\1", text)  # minor modification from ref: reduce repeating characters (>2) to only twice (=2) instead of 3 times

    # replace emoji with text strings
    if _emoji:
        # emoji_counter += emoji.emoji_count(text)
        text = emoji.demojize(text, delimiters=(' ', ' '))  # description consists of underscore: " grinning_face "
        # text = demoji.replace_with_desc(text, " ") # description " grinning face ", similar speed


    ### Stage 2: token-based processes; simple tokenization to preserve punctuation etc.

    tokens = re.split(r'([:.,!?-]+)?\s+',text)  # use white space(s) and punctuation(s) as separator but keep special characters
    tokens = list(filter(None, tokens))  # get rid of empty string '' and None

    # TODO: how to build a counter with lambda function to track the number of emoticons / acronyms being processed

    if _emoticon:
        check_emoticon = lambda x: dict_emoticon[x]['Description'] if x in set(dict_emoticon.keys()) else x
        tokens = [check_emoticon(x) for x in tokens]

    if _acronym:
        check_acronym = lambda x: dict_acronym[x.lower()]['Full'] if x.lower() in set(dict_acronym.keys()) else x
        # tokens = [check_acronym(x) for x in tokens]
        tokens = [check_acronym(re.sub(r'[^a-zA-Z\d\/]', '', x)) for x in tokens if x is not None]  # remove punctuation at the end WTF!->WTF
        tokens = list(filter(None, tokens))  # get rid of empty string '' and None


    if _spelling:
        check_spelling = lambda x: sym_spell.lookup(x, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"\w+\d")[0]._term
        tokens = [check_spelling(x) for x in tokens]


    ### Stage 3: re-based processes; need to combine tokens back to strings

    text = " ".join(tokens)

    # Spelling correction - one side effect of this spell checker is that it removes all punctuations and special characters
    # if _spelling:
        # try:
            # text = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)[0]._term
        #except IndexError:
            # text = text

    # remove all digits and numbers including 1st, 100th, etc. but excluding suffix (e.g. Windows10, Model3)
    if _numeric:
        text = re.compile('\s\d+\w+').sub(r'', text)

    # remove special characters such as ?!...;,:() in text - Not necessary when using BERT model

    if _specialChar:
        # text = re.compile(r'\W+').sub(r' ', text) # removel all non-characters
        text = re.compile(r'[_|*|^|~|&|/|\\|>|<|=]').sub(r' ', text)

    return text



### Tokenizing

def tweet_tokenizing(text, _stopwords=False, _stemming=False):
    # tokens = re.split(r'\s+',text)
    tokens = re.split(r'([:.,!?-]+)?\s+', text)
    tokens = list(filter(None, tokens))
    filtered_tokens = []

    if ((_stopwords == False) and (_stemming == False)):
       return tokens

    else:
        for token in tokens:
            # check for stop-word
            if _stopwords and nlp.vocab[token].is_stop == False:
                continue
            # check for stemming
            if _stemming:
                token = stemmer.stem(token)

            filtered_tokens.append(token)

    return filtered_tokens


### Data visualisaton

def display_stats(df_):
    # plot histogram of token counts
    fig, ax = plt.subplots(1,2,squeeze=True,tight_layout=True, figsize=(10,4))
    ax[1].hist(df_.token_lens,histtype='step')
    ax[1].set_xlabel('Token count')
    ax[1].set_ylabel('Num of Tweets')
    ax[1].set_title('Histogram of token counts \nmax length={}'.format(df_['token_lens'].max()))

    # plot daily tweets number
    ax[0].bar(df_['date'].unique(),df_.groupby('date')['text_clean'].count())
    ax[0].set_title('Number of Raw Tweets Collected by Date')
    ax[0].set_ylabel('Num of Tweets')
    ax[0].set_xlabel('Dates')
    # ax[1].set_xticklabels(labels = df_.date.unique(), rotation='vertical')
    # display(df_.groupby('date')['text_clean'].count())



# Raw tweets processing

def clean_tweets(filepath, print_stats=True, save=True, return_df = True,
                columns =['DateTime', 'Text', 'retweet_count', 'favorite_count','user.followers_count', 'tweet_id', 'user.screen_name'],
                select_columns =['tweet_id', 'DateTime', 'date', 'text_unicode', 'text_clean','tokens', 'token_lens']):

    # Loading raw data
    df_ = pd.read_csv(filepath, na_filter=False, header=None, names=columns)

    # removing duplicates in data (if any)
    df_.drop_duplicates(inplace=True)

    # converting bytes to unicode
    df_['text_unicode'] = df_['Text'].apply(lambda x: ast.literal_eval(x).decode('utf-8'))

    # set preprocessing parameters
    #global emoji_counter
    #emoji_counter = 0
    options = (True, True, True, True, True, True, True, True, False, False, False)  # spell-checker off; remove number off
    df_['text_clean'] = df_['text_unicode'].apply(tweet_preprocessing, args=options)
    df_['tokens'] = df_['text_clean'].apply(tweet_tokenizing, args=(False, False))
    df_['token_lens'] = df_['tokens'].apply(len)
    df_['date'] = pd.to_datetime(df_['DateTime']).dt.date

    # showing stats and charts
    if print_stats:
        display_stats(df_)
        display(df_[select_columns].head())
        # print(f'num of emojis processed = {emoji_counter}')
        print(f'file saved at >>> {save_file}')
        # plt.show()

     # save file
    if save:
        save_file = filepath[:-4] + '_clean.csv'
        df_[select_columns].to_csv(save_file, index=False)

    if return_df:
        return df_[select_columns]



# filepath = './Twitter_Raw/Nikola_tweets.csv'
# clean_tweets(filepath)
#
# filepath = './Twitter_Raw/Nio_tweets.csv'
# clean_tweets(filepath)
#
# filepath = './Twitter_Raw/Tesla_tweets.csv'
# clean_tweets(filepath)
#
# filepath = './Twitter_Raw/GM_tweets.csv'
# clean_tweets(filepath)
#
# filepath = './Twitter_Raw/Microsoft_tweets.csv'
# clean_tweets(filepath)
#
# # for google we have to combine files first
# path = 'C:/Users/clair/Desktop/WS2020/TA/Project/Data/Twitter_Raw/'
# files = ['Google-20201203.csv', 'Google-20201209.csv', 'Google_20201209_20201216.csv',
#          'Google_20201216_20201219.csv', 'Google_20201218_20201223.csv', 'Google_20201230_20210105.csv']
#
# all_columns = ['DateTime', 'Text', 'tweet_id', 'in_reply_to_status_id', 'retweet_count', 'favorite_count',
#                'is_quote_status','user.id_str', 'user.screen_name', 'user.verified', 'user.created_at', 'user.followers_count', 'user.friends_count',
#                'user.statuses_count', 'user.favourites_count', 'user.listed_count']
#
# select_columns = ['DateTime', 'Text', 'retweet_count', 'favorite_count', 'user.followers_count', 'tweet_id','user.screen_name']
#
# df = pd.DataFrame(columns=select_columns)
# for file in files:
#     df_ = pd.read_csv(path + file, header=None, names=all_columns)
#     df = df.append(df_[select_columns])
#     df.to_csv(path + 'Google_tweets.csv', header=False)
#
# filepath = './Twitter_Raw/Google_tweets.csv'
# clean_tweets(filepath)
#
#
# # For adding incremental raw tweets in the future
#
# def add_tweets(name,period,print_stats=False, save=True, return_df=False,
#                path = 'C:/Users/clair/Desktop/WS2020/TA/Project/Data/Twitter_Raw/'):
#
#     new_raw = name+'_'+period+'.csv'
#     master_clean = name+'_tweets_clean.csv'
#
#     df_new = clean_tweets(filepath = path+new_raw, print_stats=False, save=False, return_df=True,
#                           columns = ['DateTime', 'Text', 'tweet_id', 'in_reply_to_status_id', 'retweet_count', 'favorite_count','is_quote_status','user.id_str', 'user.screen_name', 'user.verified', 'user.created_at', 'user.followers_count', 'user.friends_count','user.statuses_count', 'user.favourites_count', 'user.listed_count'],
#                           select_columns = ['tweet_id', 'DateTime', 'date', 'text_unicode', 'text_clean','tokens', 'token_lens'])
#
#     df_master = pd.read_csv(path+master_clean,header='infer')
#     df_master_new = df_master.append(df_new).drop_duplicates(subset='tweet_id',keep='first')
#
#     if print_stats:
#         display_stats(df_master_new)
#         plt.show()
#     if save:
#         df_master_new.to_csv(path+master_clean, index=False)
#     if return_df:
#         return df_master_new
#
# period = '20210105_20210111'
# for name in ['Tesla','GM','Google','Microsoft']:
#     add_tweets(name, period, print_stats=True, save=True)
