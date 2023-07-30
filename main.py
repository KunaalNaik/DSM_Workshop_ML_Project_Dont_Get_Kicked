# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:38:57 2023

@author: USER
"""
import import_clean_columns
import preprocessing
import model_predict
import pandas as pd

# Execute the function from the script
import_clean_columns.process_data()

# Preprocessing 
train_path = 'temp/train.csv'
test_path = 'temp/test.csv'
output_train_path = 'temp/preprocessed_train.csv'
output_test_path = 'temp/preprocessed_test.csv'

train_df, test_df = preprocessing.preprocess_data(train_path, test_path)

preprocessing.write_data(train_df, output_train_path)
preprocessing.write_data(test_df, output_test_path)

# Model and Predict
# Read the data
train_df = model_predict.read_data('temp/preprocessed_train.csv')
test_df = model_predict.read_data('temp/preprocessed_test.csv')

# Select only numerical features
train_df = model_predict.select_numerical_features(train_df)
test_df = model_predict.select_numerical_features(test_df)

# Train the model and make predictions
train_predictions, test_predictions = model_predict.train_and_predict(train_df, test_df, 'isbadbuy')

# Create DataFrames for the predictions
train_predictions_df = pd.DataFrame({'RefId': train_df['refid'], 'IsBadBuy': train_predictions})
test_predictions_df = pd.DataFrame({'RefId': test_df['refid'], 'IsBadBuy': test_predictions})

# Save the predictions to CSV files
train_predictions_df.to_csv('output/train_predictions.csv', index=False)
test_predictions_df.to_csv('output/test_predictions.csv', index=False)
