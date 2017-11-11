#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:24:22 2017

@author: connorhogendorn
"""


#===============================================================
#                       IMPORT PACKAGES:
#===============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
# nltk.download() #Download the necessary datasets


#===============================================================
#                        LOAD DATA:
#===============================================================
# Create dataframe of 'raw' (as-is) categories.
train_path = 'data/train.csv'

train_df = pd.read_csv(train_path)

test_path='data/test.csv'

test_df=pd.read_csv(test_path)

X = train_df['text']
X_test = test_df['text']
y = train_df['author']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

eng_stopwords = set(stopwords.words("english"))
print('Engineering Features...')
## Number of words in the text ##
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

print('Text processing...')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#Term Frequency - Inverse Document Frequency
mod=TfidfVectorizer()
mod_TD=mod.fit_transform(X)


#SVD Features
SVD=TruncatedSVD(n_components=650,n_iter=25,random_state=24)
#Note: 400 gave overall 0.73 f1-score
#Note: 500 gave overall 0.74 f1-score, but 600 was the best... actually 650 was the best :)
SVD_FIT=SVD.fit_transform(mod_TD)
X=pd.DataFrame(SVD_FIT)

X_test = mod.transform(X_test)
X_test = SVD.transform(X_test)
X_test = pd.DataFrame(X_test)

train_df = pd.concat([train_df, X], axis=1)
test_df = pd.concat([test_df, X_test], axis=1)

### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_tfidf):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = y[dev_index], y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
train_df["nb_tfidf_char_eap"] = pred_train[:,0]
train_df["nb_tfidf_char_hpl"] = pred_train[:,1]
train_df["nb_tfidf_char_mws"] = pred_train[:,2]
test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]
test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]
test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]

cols_to_drop = ['id', 'text']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)



print('Training model...')
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_X, y,test_size=0.1,random_state=8)

clfr=XGBClassifier(base_score=0.5, 
                      colsample_bytree=0.8,
                      gamma=0, 
                      learning_rate=0.03, 
                      max_delta_step=0, 
                      max_depth=8,
                      min_child_weight=1, 
                      missing=None, 
                      n_estimators=1000, 
                      objective='multi:softprob', 
                      reg_alpha=0,
                      reg_lambda=1,
                      scale_pos_weight=1, 
                      seed=924,
                      silent=False,
                      subsample=0.8)

clfr.fit(X_train,y_train, eval_metric = "mlogloss")

# Predict:
predictions = clfr.predict(X_valid)

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_valid, predictions))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_valid, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=le.classes_,
                      title='Confusion matrix, without normalization')
plt.show()


predictions_test = clfr.predict_proba(test_X)

predictions_test = pd.DataFrame(predictions_test).reset_index(drop=True)

submission_df = pd.concat([test_df['id'],predictions_test], axis=1).reset_index(drop=True)

submission_df = submission_df.rename(columns = le_name_mapping)

submission_df.to_csv('xgb1.csv', index=False)
