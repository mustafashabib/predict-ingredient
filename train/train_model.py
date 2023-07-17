from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import json
import logging
import os
import re
import time
import unicodedata
from fractions import Fraction

import click
import joblib
import numpy as np
import openai
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from unidecode import unidecode
from word2number import w2n

# Set up OpenAI API credentials
openai.api_key = os.environ.get('OPENAI_API_KEY')

def handle_rate_limit(retries):
    wait_time = pow(2, retries)
    logging.info(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
    time.sleep(wait_time)


def generate_embeddings(texts):
    batch_size = 32  # Adjust the batch size as per the API rate limits
    embeddings = []
    num_batches = len(texts) // batch_size + \
        (1 if len(texts) % batch_size > 0 else 0)
    retries = 0
    max_retries = 3

    for i in range(num_batches):
        batch_succeeded = False
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_texts = texts[start:end]
        while not batch_succeeded and retries < max_retries:
            try:
                response = openai.Embedding.create(
                    input=batch_texts, model="text-embedding-ada-002")
                for data_obj in response['data']:
                    embeddings.append(data_obj['embedding'])
                retries = 0
                batch_succeeded = True
            except openai.error.RateLimitError:
                retries += 1
                handle_rate_limit(retries)

        if not batch_succeeded:
            raise f"Could not make API call after {retries} retries"

    return np.array(embeddings)

# Define a function to handle rate limit errors
def handle_rate_limit(retries):
    wait_time = pow(2, retries)
    logging.info(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
    time.sleep(wait_time)

# Define a function to generate embeddings using OpenAI's API
def generate_embeddings(texts):
    batch_size = 32  # Adjust the batch size as per the API rate limits
    embeddings = []
    num_batches = len(texts) // batch_size + \
        (1 if len(texts) % batch_size > 0 else 0)
    retries = 0
    max_retries = 3

    for i in range(num_batches):
        batch_succeeded = False
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_texts = texts[start:end]
        while not batch_succeeded and retries < max_retries:
            try:
                response = openai.Embedding.create(
                    input=batch_texts, model="text-embedding-ada-002")
                for data_obj in response['data']:
                    embeddings.append(data_obj['embedding'])
                retries = 0
                batch_succeeded = True
            except openai.error.RateLimitError:
                retries += 1
                handle_rate_limit(retries)

        if not batch_succeeded:
            raise f"Could not make API call after {retries} retries"

    return np.array(embeddings)


def train(data_file, ingredient_text_embeddings_file, target_ingredient_embeddings_file):
    # Load the dataset into a pandas DataFrame
    with open(data_file) as fp:
        input_data = json.load(fp)
    # Initialize lists to store the data
    ingredient_text = []
    ingredient_quantity = []
    ingredient_unit = []
    target_ingredient = []
    target_quantity = []
    target_unit = []
    data_ingredient_text_embeddings = []
    data_target_ingredient_embeddings = []
    # Process each row in the input data
    for row in input_data:
        ingredient_text.append(row["ingredient_text"])
        ingredient_quantity.append(row["ingredient_quantity"])
        ingredient_unit.append(row["ingredient_unit"])
        target_ingredient.append(row["target_ingredient"])
        target_quantity.append(row["target_quantity"])
        target_unit.append(row["target_unit"])
        
    logging.debug(f"Read data file from {data_file}")
    
    # Preprocess the text features using OpenAI embeddings
    if not os.path.exists(ingredient_text_embeddings_file):
        logging.debug(
            f"Generating ingredient text embeddings and saving to {ingredient_text_embeddings_file}")
        ingredient_text_embeddings = generate_embeddings(ingredient_text)
        np.save(ingredient_text_embeddings_file, ingredient_text_embeddings)
    else:
        logging.debug(
            f"Loading ingredient text embeddings from {ingredient_text_embeddings_file}")
        ingredient_text_embeddings = np.load(ingredient_text_embeddings_file)

    data_ingredient_text_embeddings = ingredient_text_embeddings

    # Preprocess the text features using OpenAI embeddings
    if not os.path.exists(target_ingredient_embeddings_file):
        logging.debug(
            f"Generating target ingredient embeddings and saving to {target_ingredient_embeddings_file}")
        target_ingredient_embeddings = generate_embeddings(target_ingredient)
        np.save(target_ingredient_embeddings_file, target_ingredient_embeddings)
    else:
        logging.debug(
            f"Loading target ingredient embeddings from {target_ingredient_embeddings_file}")
        target_ingredient_embeddings = np.load(target_ingredient_embeddings_file)

    data_target_ingredient_embeddings = target_ingredient_embeddings

    # Create a new DataFrame
    data = pd.DataFrame({
        "ingredient_quantity": ingredient_quantity,
        "ingredient_unit": ingredient_unit,
        "target_ingredient": target_ingredient,
        "target_quantity": target_quantity,
        "target_unit": target_unit,
    })

    data["ingredient_text_embedding"] = data_ingredient_text_embeddings
    data["target_ingredient_embedding"] = data_target_ingredient_embeddings
    # Encode the categorical features using OneHotEncoder
    logging.debug(
        "One hot encoding ingredient unit, target unit, target ingredient")
    # Encode the categorical features using OneHotEncoder
    encoder_ingredient_unit = OneHotEncoder(sparse=False)
    encoder_target_unit = OneHotEncoder(sparse=False)

    ingredient_unit_encoded = encoder_ingredient_unit.fit_transform(data[["ingredient_unit"]])
    target_unit_encoded = encoder_target_unit.fit_transform(data[["target_unit"]])
    # Convert the encoded arrays into DataFrames
    ingredient_unit_encoded_df = pd.DataFrame(ingredient_unit_encoded, columns=encoder_ingredient_unit.categories_)
    target_unit_encoded_df = pd.DataFrame(target_unit_encoded, columns=encoder_target_unit.categories_)
    # Concatenate all columns into a new DataFrame
    data_encoded = pd.concat([data.drop(["ingredient_unit", "target_unit"], axis=1),
                              ingredient_unit_encoded_df, target_unit_encoded_df], axis=1)


    # Split the dataset into training and testing sets
    X = data_encoded[["ingredient_text_embedding", "ingredient_unit_encoded"]]
    y = data_encoded[["target_ingredient_embedding", "target_quantity"]]
    # Convert column names to strings
    X.columns = X.columns.astype(str)
    y.columns = y.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=92080)


    # Initialize the base estimator and the multi-output regressor
    base_estimator = RandomForestRegressor()
    model = MultiOutputRegressor(base_estimator)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


@click.command(help='Provided a data-file in the correct form, the location of your embeddings files (or where to create them), and train a model to predict ingredient matches')
@click.argument('data_file', required=True, type=click.Path(exists=True))
@click.argument('ingredient_embeddings_file', required=True)
@click.argument('target_ingredient_embeddings_file', required=True)
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO', help='Logging level')
def main(data_file, ingredient_embeddings_file, target_ingredient_embeddings_file, log_level):
    # Configure logger
    logging.basicConfig(level=log_level)
    train(data_file, ingredient_embeddings_file,
          target_ingredient_embeddings_file)


if __name__ == '__main__':
    main()
