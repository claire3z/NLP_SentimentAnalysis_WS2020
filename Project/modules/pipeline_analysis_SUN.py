''"""
Created by: Claire Z. Sun
Date: 2021.02.04
''"""

###################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import pickle
import time
import torch
import transformers
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from wordcloud import WordCloud
import gc

# in case of reloading modules after modification
#import sys
#import importlib
#importlib.reload(sys.modules['modules.preprocessing_SUN'])

use_GPU = True
path = 'C:/Users/clair/Desktop/WS2020/NLP/Project/'
# file paths to saved models
finetuned_save = path+"models/bertbase_state.bin"
nb_classifier_save = path+'models/Count_word_NB_balancedWeight.pickle'
count_vectorizer_save = path+'models/CountVectorizer_unigram.pickle'
lr_classifier_save = path+'models/Tfidf_word_LR_balancedWeight.pickle'
tfidf_vectorizer_save = path+'models/TfidfVectorizer_unigram.pickle'
svm_classifier_save =path+'models/Word2Vec_news300_aggMean_SVM_balancedWeight.pickle'
w2v_vectorizer_save = path+'models/Word2Vec_news300_aggMean.pickle'
price_rnn_save = path+'models/RNN_state_v2.bin'

# set up device
if (use_GPU and torch.cuda.is_available()):
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('Using CPU')
    device = torch.device("cpu")

# load customized functions
from modules.preprocessing_SUN import tweet_preprocessing, tweet_tokenizing, display_stats
from modules.vader_SUN import vader
from modules.ML_classifiers_SUN import ML_classifier_vectorizer
from modules.BERT_finetuning_SUN import TwitterRawDataset, SentimentClassifier_BERT,infer_raw, bert_base
from modules.price_prediction_SUN import RNN

#from modules.ensemble_SUN import Ensemble, Ensemble2

# download pretrained BERT model
pretrained_bert = transformers.BertModel.from_pretrained('bert-base-cased')
tokenizer_bert = transformers.BertTokenizer.from_pretrained('bert-base-cased')

# dictionary to translate sentiments (str) to class labels (int)
sentiment_to_label = {'negative':0,'neutral':1,'positive':2}
sentiments = ['negative','neutral','positive']

def run_sentiment_analysis(ticker, files, sentiments_save, predictions_save, log_save, vis_save):
    """
    :param ticker: str - name of the stock to be displayed on charts and log file
    :param files: list - containing filename or filenames of raw tweets collected
    :param sentiments_save: str .csv - saving sentiment summary (daily counts of pos, neg, neu sentiment by majority vote)
    :param predictions_save: str .csv - saving predictions of all tweets by different models
    :param log_save: str .txt - saving processing time log
    :param vis_save: str .png - saving data visualisation
    :return: None
    """
    start_ = time.time()
    log = open(log_save, "a")  # append mode
    log.write(f'### {ticker} ###')
    print(f'### {ticker} ###')
    # load raw tweets - should be tweets on the same stock
    print(f'\nLoading raw tweets from {len(files)} files:')
    log.write(f'\n\nLoading raw tweets from {len(files)} files:')
    df = pd.DataFrame(columns=['DateTime', 'Text', 'ID'])
    start_time = time.time()
    for file in files:
        df_ = pd.read_csv(file, na_filter=False, header=None).iloc[:, 0:3]
        df_.columns = ['DateTime', 'Text', 'ID']
        df = df.append(df_)
    print(f'>> loading: {time.time() - start_time:.3f} s')
    log.write(f'\n>> loading: {time.time() - start_time:.3f} s')
    # remove duplicated tweets (due to downloads on overlapping days)
    df.drop_duplicates(subset='ID', inplace=True)  # len(df)

    # text preprocessing
    print(f'\nStart processing {len(df)} raw tweets:')
    log.write(f'\n\nStart processing {len(df)} raw tweets:')
    start_time = time.time()
    df['date'] = pd.to_datetime(df['DateTime']).dt.date
    df['text_unicode'] = df['Text'].apply(lambda x: ast.literal_eval(x).decode('utf-8'))
    df['text_clean'] = df['text_unicode'].apply(tweet_preprocessing)
    df['token_lens'] = df['text_clean'].apply(tweet_tokenizing, args=(False, False)).apply(
        len)  # no stemming and no stopwords removal
    print(f'>> text preprocessing: {time.time() - start_time:.3f} s')
    log.write(f'\n>> text preprocessing: {time.time() - start_time:.3f} s')
    # remove columns no longer needed and reset index
    df.drop(['DateTime', 'Text', 'ID'], axis=1, inplace=True)
    df = df.reset_index(drop=True)

    # sentiments prediction - comparing speed of different models
    print(f'\nStart sentiment analysis on {len(df)} tweets:')
    log.write(f'\n\nStart sentiment analysis on {len(df)} tweets:')
    start_time = time.time()
    df['vader'] = df['text_clean'].apply(vader)
    print(f'>> vader: {time.time() - start_time:.3f} s')
    log.write(f'\n>> vader: {time.time() - start_time:.3f} s')

    start_time = time.time()
    df['nb_count'] = ML_classifier_vectorizer(df['text_clean'], nb_classifier_save, count_vectorizer_save)
    print(f'>> nb_count: {time.time() - start_time:.3f} s')
    log.write(f'\n>> nb_count: {time.time() - start_time:.3f} s')

    start_time = time.time()
    df['lr_tfidf'] = ML_classifier_vectorizer(df['text_clean'], lr_classifier_save, tfidf_vectorizer_save)
    print(f'>> lr_tfidf: {time.time() - start_time:.3f} s')
    log.write(f'\n>> lr_tfidf: {time.time() - start_time:.3f} s')

    start_time = time.time()
    df['svm_w2v'] = ML_classifier_vectorizer(df['text_clean'], svm_classifier_save, w2v_vectorizer_save)
    print(f'>> svm_w2v: {time.time() - start_time:.3f} s')
    log.write(f'\n>> svm_w2v: {time.time() - start_time:.3f} s')

    start_time = time.time()
    df['bert_base'] = bert_base(df['text_clean'], pretrained_bert, tokenizer_bert, finetuned_save, device,
                                batch_size=256)  # len(df): CUDA out of memory 5.35G
    print(f'>> bert: {time.time() - start_time:.3f} s')
    log.write(f'\n>> bert: {time.time() - start_time:.3f} s ({device})')

    start_time = time.time()
    df['majority'] = df[['bert_base', 'vader', 'nb_count', 'lr_tfidf', 'svm_w2v']].mode(axis=1)[0].astype(int)
    print(f'>> majority: {time.time() - start_time:.3f} s')
    log.write(f'\n>> majority: {time.time() - start_time:.3f} s')

    # saving files
    selected = ['majority', 'bert_base', 'svm_w2v', 'lr_tfidf', 'nb_count', 'vader']
    df[['date', 'text_unicode', 'token_lens'] + selected].to_csv(predictions_save, index=False)

    # separate daily predictions
    df_summary = pd.get_dummies(df[['date', 'majority']], columns=['majority']).groupby('date').sum()
    df_summary.sort_values(by='date').to_csv(sentiments_save)  # keep index as date

    print(f"\nSentiment analysis completed! Total processing time: {(time.time() - start_) / 60:.1f} min\n")
    log.write(
        f"\n\nSentiment analysis for {ticker} completed! Total processing time: {(time.time() - start_) / 60:.1f} min\n")
    log.close()

    # data visualisation
    fig, ax = plt.subplots(2, 2, figsize=(16, 8), tight_layout=True)
    fig.suptitle(f"{ticker} Overview", fontsize=14)

    # word cloud
    text = df['text_clean'].to_string().replace('USERNAME', '')
    wordcloud = WordCloud(max_font_size=40).generate(text)
    ax[0, 0].imshow(wordcloud, interpolation='bilinear')
    ax[0, 0].axis("off");
    ax[0, 0].set_title('Top Words')

    # Distribution of token lengths per tweet
    ax01 = sns.distplot(df['token_lens'], ax=ax[0, 1])
    ax01.set_xlabel('Token counts')
    ax01.set_ylabel('Num of Tweets')
    ax01.set_title('Tweet Lengths (max={})'.format(df['token_lens'].max()))

    # Daily Tweets sentiment distribution
    ind = df_summary.index
    cmap = sns.color_palette("tab10", 3)  # same color scheme as default matplotlib
    ax[1, 0].bar(ind, df_summary['majority_0'])
    ax[1, 0].bar(ind, df_summary['majority_1'], bottom=df_summary['majority_0'])
    ax[1, 0].bar(ind, df_summary['majority_2'], bottom=df_summary['majority_0'] + df_summary['majority_1'])
    ax[1, 0].set_title('Daily Tweets Sentiment Distribution')
    ax[1, 0].set_ylabel('Num of Tweets collected per day')

    # Prediction comparison by different models
    ax11 = sns.heatmap(df[selected].sort_values(by='majority', ascending=False), cmap=cmap, yticklabels=False,
                       ax=ax[1, 1])
    ax11.set_title('Prediction Discrpency Overview')
    colorbar = ax11.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.set_ticklabels(sentiments)
    plt.savefig(vis_save)
    plt.show()


def run_price_prediction(company, price_save,sentiments_save):
    # share price aggregation
    df_price = pd.read_csv(price_save, header='infer')
    df_price = df_price[['Date', "Adj Close"]].rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df_price['chg_1d'] = df_price["adj_close"].diff(1) / df_price["adj_close"]

    # sentiment aggregation
    df_sentiment = pd.read_csv(sentiments_save, header='infer')
    df_sentiment['total'] = df_sentiment[['majority_0', 'majority_1', 'majority_2']].sum(axis=1)
    df_sentiment[['negative', 'neutral', 'positive']] = df_sentiment[['majority_0', 'majority_1', 'majority_2']].div(df_sentiment['total'], axis=0)
    df_sentiment[['chg_positive', 'chg_neutral', 'chg_negative']] = df_sentiment[['negative', 'neutral', 'positive']].diff(1)
    df_sentiment['num'] = 0.5  # (sentiment_TSLA['total'] - sentiment_TSLA['total'].min()) / (sentiment_TSLA['total'].max() - sentiment_TSLA['total'].min())

    # merge share price with sentiment
    df_ = df_price[['date', 'chg_1d']].merge(df_sentiment[['date', 'num', 'chg_positive', 'chg_neutral', 'chg_negative']], on='date', how='outer', sort=True)

    # aggregate non-trading day sentiment to last trading day
    for i in range(len(df_) - 1, 1, -1):
        if np.isnan(df_['chg_1d'].iloc[i]):
            df_.loc[i - 1, ['num', 'chg_positive', 'chg_neutral', 'chg_negative']] = (
                    df_[['num', 'chg_positive', 'chg_neutral', 'chg_negative']].iloc[i - 1] +
                    df_[['num', 'chg_positive', 'chg_neutral', 'chg_negative']].iloc[i])
    df_.dropna(inplace=True)
    df_.reset_index()  # .to_csv('output/' + name + '_processed.csv')

    lag = 1
    features = ['chg_1d', 'chg_positive', 'chg_neutral', 'chg_negative', 'num']
    h_dim = 5  # number of hidden states in the RNN/LSTM models
    num_layers = 1  # number of layers in the RNN/LSTM models

    # load price prediction model
    model = RNN(input_size=len(features), hidden_size=h_dim, output_size=lag, num_layers=num_layers)
    model.load_state_dict(torch.load(price_rnn_save))
    model.eval()

    # model = CNN(numFeatures=len(features),numLags=lag,numHidden=h_dim)
    # model.load_state_dict(torch.load(price_cnn_save))
    # model.eval()

    data = df_[features].to_numpy()
    x = torch.Tensor(data.reshape(1, 1, -1))
    pred = model(x)

    pred_chg = pred.detach().numpy().item()
    pred_price = (1 + pred_chg) * df_price.loc[len(df_price) - 2, ['adj_close']].item()
    df_price[['pred_chg', 'pred_price']] = np.nan
    df_price.loc[len(df_price) - 1, ['pred_chg']] = pred_chg
    df_price.loc[len(df_price) - 1, ['pred_price']] = pred_price
    df_price.loc[len(df_price) - 2, ['pred_price']] = df_price.loc[len(df_price) - 2, ['adj_close']].item()

    # visualization
    fig, ax = plt.subplots(2, 1, figsize=(10, 8),sharex=True)
    fig.suptitle(f'Share Price Forecast for {company}')
    ax[0].set_title(f'Share Price')
    ax[0].plot(df_price['date'], df_price['pred_price'], label='predicted', color='C1', marker='o', linestyle='dashed')
    ax[0].plot(df_price['date'], df_price['adj_close'], color='C0', marker='o', label='actual')
    ax[0].legend()

    width = 0.35
    x = np.arange(1,len(df_price))
    ax[1].set_title(f'Pct Chg. vs Previous day')
    ax[1].bar(x+width/2, df_price['chg_1d'], width, label='actual')
    ax[1].bar(x-width/2, df_price['pred_chg'], width, label='predicted')
    ax[1].set_xticklabels(df_price['date'])
    ax[1].axhline(y=0, c='k', linewidth=0.5)
    plt.savefig(f'output/{company}_price_prediction.png')
    plt.show()

    print(f'Predicted next day price movement for {company}: {pred_chg*100:.2f}%')
    print(f'Predicted next day closing share price: ${pred_price:.2f}/share')

