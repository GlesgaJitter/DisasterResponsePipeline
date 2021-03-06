# import libraries
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import time
import pickle
#import LinearSVC


import sys


def load_data(database_filepath):
    """
    INPUT:
    - database_filepath: string - location of database to be loaded
    
    OUTPUT:
    - X: List of messages extracted from database
    - y: DataFrame of category columns
    - category_names: list of names of category columns
    
    This function loads the cleaned database (saved in ../data/process_data.py) 
    and splits it into the two outputs.
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_cleaned', 'sqlite:///'+database_filepath)
    X = df.message.values
    y = df.drop(columns=['id','message','original','genre'])
    catgeory_names = y.columns.tolist()
    
    return X, y, catgeory_names


def tokenize(text):
    """
    # INPUT: text - string of text to be tokenised

    # OUTPUT: clean_tokens - list of tokenised words
    
    This functions takes messages as sentnces and returns a 
    list of cleaned and lemmatised tokens. 
    """

    # make lower case and remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenise and lemmatise
    words = word_tokenize(text)
    #words = [w for w in words if w not in stopwords.words("english")]
    lemmatiser = WordNetLemmatizer()

    clean_tokens = []
    for word in words:
        clean_word = lemmatiser.lemmatize(word).lower().strip()
        clean_tokens.append(clean_word)

    return clean_tokens


def build_model():
    """
    INPUT:
    - None
    
    OUTPUT: 
    - cv: GridSearch over machine learning pipeline
    
    This function builds the ML model using the classifier
    - ExtraTreesClassifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
    ])

    parameters = {
            'clf__estimator__max_features': ['auto', 'sqrt', 'log2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    #pipeline.fit(X_train, y_train)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT: 
    - model: machine learning model
    - X_test: partition of messages to be used to test ML model
    - Y_test: partition of categories to be used to test ML model
    - category_names: list of category column names
    
    OUTPUT:
    - None - function prints classification report to evaluate ML model
    
    This function evaluates the performance of the ML model and prints a 
    classification report
    """
    # predict on test data
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=y_test.columns))

def save_model(model, model_filepath):
    """
    INPUT: 
    - model: ML model
    - model_filepath: string - pickle file name for ML model. Provided by user
    
    OUTPUT:
    - None: function creates a .pkl to save the ML model
    
    This function saves the machine learning model as a pickle file
    """
    with open('model_file.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
