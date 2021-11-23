from typing import Text
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Create flask app

flaskApp = Flask(__name__)

@flaskApp.route('/', methods = ['GET','POST'])
def home():
    return 'Home UI'

@flaskApp.route('/rating', methods = ['GET','POST'])
def rating():

    #load the vectorizer
    with open('vectorizer','rb') as f:
        loaded_CV = pickle.load(f)

    #load the trained models
    with open('logReg_pickle','rb') as f:
        lr = pickle.load(f)
    with open('sgd_pickle','rb') as f:
        sg = pickle.load(f)

    recievedJsonFile = request.json

    recievedText = recievedJsonFile["reviewText"]
    inputText = [recievedText]

    #inputText = []
    
    #fit input text to model
    inputFeatures = loaded_CV.transform(inputText)

    #choose the correct model here
    predictions = lr.predict(inputFeatures)

    #send results back
    result = str(predictions[0])
    
    #return jsonify({"prediction":result})
    return result

@flaskApp.route('/tags', methods = ['GET','POST'])
def tags():
    
    #load the tfidf vectorizer
    with open('TCvectorizer','rb') as f:
        tfidf = pickle.load(f)

    #load the classifier
    with open('TC_pickle','rb') as f:
        clf = pickle.load(f)

    #load the binarizer
    with open('multiLabelBiner','rb') as f:
        multilabel = pickle.load(f)

    #get the user input
    recievedJsonFile = request.json

    x = [recievedJsonFile["reviewText"]]
    print(x)

    #classification with input
    #x = [ 'good for your children and crowded and fun']

    vectors = tfidf.transform(x)
    clf.predict(vectors)

    tagsSet = multilabel.inverse_transform(clf.predict(vectors))
    print(tagsSet[0])
    

    #return tagsSet
    return jsonify(tagsSet[0])

    #respective tags :  'calm', 'child', 'clean', 'crowded', 'family', 'food', 'fun'


         

if __name__ == "__main__":
    flaskApp.run()
