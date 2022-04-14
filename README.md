# Disaster Response Pipeline Project

This program trains an ML model to classify disaster response messages, 
and runs a Flask app in which users can test disaster messages to see their classification. 

This program has two main parts:
- Machine learning programme
We train an ML model using a sample messages dataset, which includes the columns
id | message | original | genre
and a corresponding disaster categories dataset, with columns
id | categories
We combine and clean these tables, and save locally a database which we call in the model training step. 
We train a multi-category ML model, using RandomForestClassifier from sklearn, and save this model as a pickle file. 
- Flask app
We include an interactive Flask app which presents visuals o the training data, and a free-format message classifier. 


This project comprises the below files
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py  # file to process and clean input data
|- messages_cleaned.db   # database to save clean data to

- models
|- train_classifier.py  # file to train ML classification model
|- classifier.pkl  # saved model 

- README.md  # this file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
