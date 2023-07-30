# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:47:42 2023

@author: USER
"""
# Importing necessary libraries
import pandas as pd
import os

def process_data():
    # Define the paths
    input_path = 'input/'
    output_path = 'temp/'

    # Reading the training and test data from CSV files
    train = pd.read_csv(os.path.join(input_path, 'training.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))

    # Converting all column names to lower case
    train.columns = map(str.lower, train.columns)
    test.columns = map(str.lower, test.columns)

    # Exporting the dataframes to new CSV files in the 'temp' folder
    train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=False)

if __name__ == "__main__":
    process_data()
