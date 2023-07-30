# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:03:49 2023

@author: USER
"""
import pandas as pd
import numpy as np
#import os

def read_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing_values(df):
    # Fill missing values in numeric columns with the column median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column].fillna(df[column].median(), inplace=True)

    # Fill missing values in date columns with the earliest date
    date_columns = df.select_dtypes(include=['datetime']).columns
    for column in date_columns:
        df[column].fillna(df[column].min(), inplace=True)

    # Fill missing values in categorical columns with the most frequent category
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    return df


def handle_outliers(df):
    # Handle outliers as necessary. This is just a placeholder.
    # You might use methods like the Z-score or the IQR method to handle outliers.
    return df

def preprocess_data(train_path, test_path):
    # Read the data
    train_df = read_data(train_path)
    test_df = read_data(test_path)
    
    # Handle missing values
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # Handle outliers
    train_df = handle_outliers(train_df)
    test_df = handle_outliers(test_df)

    return train_df, test_df

def write_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    train_path = 'temp/train.csv'
    test_path = 'temp/test.csv'
    output_train_path = 'temp/preprocessed_train.csv'
    output_test_path = 'temp/preprocessed_test.csv'

    train_df, test_df = preprocess_data(train_path, test_path)

    write_data(train_df, output_train_path)
    write_data(test_df, output_test_path)
