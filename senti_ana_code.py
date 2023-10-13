import urllib.request
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk


nltk.download("stopwords")

data=pd.read_csv("https://confrecordings.ams3.digitaloceanspaces.com/amazon_alexa.tsv",delimiter='\t')
 


#print(data.head())

tokenizer=RegexpTokenizer(r'\w+')

en_stopwords=set(stopwords.words('english'))

ps=PorterStemmer()

def getStemmedReview(review):

    #review=review.lower()

    review=review.replace("<br /><br />"," ")

    #Tokenize

    tokens=tokenizer.tokenize(review)

    new_tokens=[token for token in tokens if token not in  en_stopwords]

    stemmed_tokens=[ps.stem(token) for token in new_tokens]

    clean_review=' '.join(stemmed_tokens)

    return clean_review



data['verified_reviews']=data['verified_reviews'].apply(getStemmedReview)



X=data['verified_reviews']

Y=data['feedback']



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',decode_error='ignore')

vectorizer.fit(x_train)

x_train=vectorizer.transform(x_train)

x_test=vectorizer.transform(x_test)



from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='liblinear')

model.fit(x_train,y_train)



print("Score on training data is: "+str(model.score(x_train,y_train)))

print("Score on testing data is: "+str(model.score(x_test,y_test)))

import joblib

joblib.dump(en_stopwords,'stopwords.pkl') 

joblib.dump(model,'model.pkl')

joblib.dump(vectorizer,'vectorizer.pkl')

loaded_model=joblib.load("model.pkl")

loaded_stop=joblib.load("stopwords.pkl")

loaded_vec=joblib.load("vectorizer.pkl")

def classify(text_box_1):

 label = {0: 'negative', 1: 'positive'}

 X = loaded_vec.transform([text_box_1])

 y = loaded_model.predict(X)[0]

 proba = np.max(loaded_model.predict_proba(X))

 if proba > 0.85:

   return 'positive'

 else:

   return 'negative'


   

print(classify('This alexa Echo is awesome . I recommend you also to purchase this product.'))

print(classify('I like this product very much.'))

print(classify('I do not like any product of alexa.'))

print(classify('It’s sound is loud , clear , free from echo and pleasant.'))

print(classify('I had complained about this product numerous time but get no reply.'))

