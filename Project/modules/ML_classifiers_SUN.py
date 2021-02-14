import numpy as np
import pandas as pd
import pickle
import re
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, f1_score


### Helper functions

#import gensim.downloader as api
#model_glove_twitter = api.load("glove-twitter-25")
#model_glove_twitter = api.load("glove-twitter-200")
#vectorizer = WordEmbeddingVectorizer(model_glove_twitter,mode='aggregate',by='mean')

class WordEmbeddingVectorizer():
    "Wrapper around pre-trained word-embedding model"
    def __init__(self, model, mode, by):
        self.model = model
        self.dim = model.vector_size
        self.mode = mode
        self.by = by

    def transform(self, data):
        data_ = data.apply(lambda x: re.split('\s+', x)).apply(vectorize,args=(self.model, self.dim, self.mode, self.by))
        data_ = np.array(data_.to_list())  # array
        #scaler = MinMaxScaler()  # rescale to 0-1, NB only works for non-negative values
        #vect = scaler.fit_transform(data_)  # ndarray array of shape (n_samples, n_features_new)
        return data_ #vect



def vectorize(words, emb, d, mode=None, by='mean', max_len=100):
    '''vectorize a list of words into vectors according to word embeddings
    words: a list of words
    emb: pre-trained word-embeddings
    d: dimentionality of embeddings
    mode: default None, optional "aggregate" or 'concatenate',
    by: 'mean', 'sum' or 'max'
    max_len: padding to max_len
    '''

    if mode == None:
        a = np.zeros((max_len, d), dtype=float)
        for i in range(len(words)):
            w = words[i]
            if w in emb.vocab.keys():
                a[i] = emb[w]
            return a

    if mode == 'aggregate':
        a = np.zeros((len(words), d), dtype=float)
        for i in range(len(words)):
            w = words[i]
            if w in emb.vocab.keys():
                a[i] = emb[w]
        if by == 'mean':
            return a.mean(axis=0)
        if by == 'max':
            return a.max(axis=0)
        if by == 'sum':
            return a.sum(axis=0)

    if mode == 'concatenate':
        a = np.zeros((max_len, d), dtype=float)
        for i in range(len(words)):
            w = words[i]
            if w in emb.vocab.keys():
                a[i] = emb[w]
            return a.flatten()


def train_classifier(clf, x_train_res, y_train_res, x_valid_vect, y_valid, model_save, report_save):
    """
    clf: sklearn classifier or classifier pipeline with scaler
    x_train_res: resampled (e.g. SMOTE) vectorized training data
    y_train_res: resampled (e.g. SMOTE) training label
    x_valid_vect: vectorized data for validation
    y_valid: validation label
    model_save: str
    report_save: str
    """

    time_start = time.time()

    # fit and transform training data on classifier
    clf.fit(x_train_res, y_train_res)
    time_train = time.time()

    # make predictions
    y_pred_train = clf.predict(x_train_res)
    y_pred_valid = clf.predict(x_valid_vect)
    time_predict = time.time()

    report_train = classification_report(y_train_res, y_pred_train, output_dict=True)
    print("\nTraining Set Performance:\n", classification_report(y_train_res, y_pred_train))

    report_valid = classification_report(y_valid, y_pred_valid, output_dict=True)
    print("\nValidation Set Performance:\n", classification_report(y_valid, y_pred_valid))

    # comparing speed
    print(f"Time to train: {time_train - time_start: .2f}s")
    print(f"Time to predict: {time_predict - time_train: .2f}s")

    # saving model - classifier with scaler
    with open(model_save, 'wb') as picklefile:
        pickle.dump(clf, picklefile)

    # saving classification reports
    df_report_train = pd.DataFrame(report_train).transpose()
    df_report_train['report'] = 'train'
    df_report_valid = pd.DataFrame(report_valid).transpose()
    df_report_valid['report'] = 'valid'
    df_report_train.append(df_report_valid).to_csv(report_save)




def run_classification(vectorizer, classifier, x_train, y_train, x_test, y_test, model_save, report_save):
    # vectorize training data
    time_start = time.time()

    try:
        x_train_vect = vectorizer.transform(x_train).toarray()  # from sparse matrix to array
    except AttributeError:
        x_train_vect = vectorizer.transform(x_train)  # already ndarray

    # resample training data to deal with class imbalance
    smote = SMOTE()
    x_train_res, y_train_res = smote.fit_sample(x_train_vect, y_train)
    time_vect_train = time.time()

    # standardize data and make pipeline
    clf = make_pipeline(StandardScaler(), classifier)

    # fit and transform training data on classifier
    clf.fit(x_train_res, y_train_res)
    time_train = time.time()

    y_pred_train = clf.predict(x_train_res)
    report_train = classification_report(y_train_res, y_pred_train, output_dict=True)
    print("\nClassification Report for Training Set:\n", classification_report(y_train_res, y_pred_train))

    # make predictions on test set
    time_start_test = time.time()
    x_test_vect = vectorizer.transform(x_test)
    time_vect_test = time.time()

    y_pred_test = clf.predict(x_test_vect)
    time_pred_test = time.time()
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    print("\nClassification Report for Validation Set:\n", classification_report(y_test, y_pred_test))

    # comparing speed of prediction
    print(f"training_vectorization: {time_vect_train - time_start: .2f}s")
    print(f"training_train time: {time_train - time_vect_train: .2f}s")

    print(f"validation_vectorization: {time_vect_test - time_start: .6f}s")
    print(f"validation_prediction time: {time_pred_test - time_vect_test: .6f}s")

    # saving model
    with open(model_save, 'wb') as picklefile:
        pickle.dump(clf, picklefile)

    # saving classification reports
    df_report_train = pd.DataFrame(report_train).transpose()
    df_report_train['report'] = 'train'
    df_report_test = pd.DataFrame(report_test).transpose()
    df_report_test['report'] = 'valid'
    df_report_train.append(df_report_test).to_csv(report_save)



### Naive Bayes Classifier with CountVectorizer
def NB_countVect_classifier(text_clean):

    with open('./models/CountVectorizer_unigram.pickle','rb') as f:
        vectorizer = pickle.load(f)

    with open('./models/NB_countVect_unigram.pickle','rb') as f:
        classifier = pickle.load(f)

    x_vect = vectorizer.transform(text_clean)

    return classifier.predict(x_vect)



### LR with TfidfVectorizer
def LR_tfidf_classifier(text_clean):

    with open('./models/TfidfVectorizer_unigram.pickle','rb') as f:
        vectorizer = pickle.load(f)

    with open('./models/LR_TfidfVect_unigram.pickle','rb') as f:
        classifier = pickle.load(f)

    x_vect = vectorizer.transform(text_clean)

    return classifier.predict(x_vect)



### SVM with GloVe embeddings 25D Twitter
def SVM_GloVe25_classifier(text_clean):
    # take a long time to load
    with open('./models/GloVe_Twitter_25.pickle','rb') as f:
        vectorizer = pickle.load(f)

    with open('./models/SVM_GloVe_Twitter_25_aggMean.pickle','rb') as f:
        classifier = pickle.load(f)

    x_vect = vectorizer.transform(text_clean)

    return classifier.predict(x_vect)



### LR with W2V embeddings 300D Google news
def LR_w2v300_classifier(text_clean):
    # take a long time to load
    with open('./models/w2v_news_300.pickle','rb') as f:
        vectorizer = pickle.load(f)

    with open('./models/LR_w2v_news_300_aggMean.pickle','rb') as f:
        classifier = pickle.load(f)

    x_vect = vectorizer.transform(text_clean)

    return classifier.predict(x_vect)


sentiment_to_label = {'negative':0,'neutral':1,'positive':2}

def ML_classifier_vectorizer(df_text,classifier_save, vectorizer_save):
    with open(vectorizer_save, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(classifier_save, 'rb') as f:
        classifier = pickle.load(f)
    x_vect = vectorizer.transform(df_text)
    predictions = classifier.predict(x_vect) #output sentiment
    return pd.DataFrame(predictions).replace(sentiment_to_label)


