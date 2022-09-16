from wrangle import gdb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#~~~~~~~~~~~~~~~~~~~~~~~~~<   >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def split_data_discrete(df, strat_by, rand_st=123):
    '''
    Takes in: a pd.DataFrame()
          and a column to stratify by  ;dtype(str)
          and a random state           ;if no random state is specifed defaults to [123]
          
      return: train, validate, test    ;subset dataframes
    '''
    # from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=.2, 
                               random_state=rand_st, stratify=df[strat_by])
    train, validate = train_test_split(train, test_size=.25, 
                 random_state=rand_st, stratify=train[strat_by])
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')


    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~<   >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def prep_iris():
    df = get_iris_data()
    df.drop(columns=['measurement_id', 'species_id', 'species_id.1'], inplace=True)
    df.rename(columns = {'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(df.species, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    print(df.info())
    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~<   >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def get_iris_data():
    filename = "iris.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = gdb('iris_db', 
                '''
                        SELECT * FROM measurements m
			JOIN species s
            ON s.species_id = m.species_id;
                ''')
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df