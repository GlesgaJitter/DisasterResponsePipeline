import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT: 
    - messages_filepath: string - location of messages file
    - categories_filepath: string - location of categories file
    
    OUTPUT:
    - df: DataFrame - frame of joined (on message id) messages and categories
    
    This function loads and merges the messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='inner', left_on='id', right_on='id')

    return df

def clean_data(df):
    """
    INPUT:
    - df: DataFrame of messages and associated categories
    
    OUTPUT:
    - df: DataFrame of messages and categories in separate columns
    where '1' or '0' indicate a message does or does not fall into 
    a category respectively
    
    This function cleans the categories column of the merged DataFrame. 
    The categories column is split into 36 separate columns and fills these
    new columns with '0' or '1' to indicate a message falls into a category.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        categories.replace(to_replace=2, value=1, inplace=True)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    INPUT:
    - df: DataFrame of messages and associated categories
    - database_filename: string - name given to database. Provided by user
    
    OUTPUT: 
    - None - function creates a database file
    
    This function saves the cleaned DataFrame as an SQL database. 
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_cleaned', engine, index=False, if_exists='replace')


def main():
    """
    INPUT: 
    - None: this main function takes no inputs
    
    OUTPUT:
    - None: this main function returns no outputs
    
    This function takes user inputs and runs the above functions to:
    - Load data
    - Clean data
    - Save data
    or
    - print a user-input error
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

#python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
if __name__ == '__main__':
    main()
