import json
import logging
import os
import re
import time
import unicodedata

import click
import joblib
import numpy as np
import openai
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from word2number import w2n

# Set up OpenAI API credentials
openai.api_key = os.environ.get('OPENAI_API_KEY')


def extract_quantity_and_unit(text):
    # Define common units used in recipe ingredients
    common_units = [
        'cup', 'cups', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons',
        'ounce', 'ounces', 'pound', 'pounds', 'gram', 'grams', 'kilogram', 'kilograms',
        'ml', 'milliliter', 'milliliters', 'liter', 'liters', 'gallon', 'gallons',
        'pinch', 'pinches', 'dash', 'dashes', 'slice', 'slices', 'cloves', 'clove'
    ]
    # Regular expressions to extract quantity and unit
    quantity_regex = r"(\d+(?:\.\d+)?|\d*\s*[\u00BC-\u00BE\u2150-\u215E])"
    unit_regex = r'({})'.format('|'.join(common_units))

    # Extract quantity
    quantity = None
    unit = None

    # Extract quantity and unit from the first match
    quantity_match = re.search(quantity_regex, text)
    if quantity_match:
        quantity_text = quantity_match.group(1)
        # Handle numeric values
        if quantity_text.isdigit() or quantity_text[0].isdigit():
            quantity = float(quantity_text)
        # Handle decimal values
        elif quantity_text.startswith('.'):
            quantity = float('0' + quantity_text)
        elif quantity_text.endswith('.'):
            quantity = float(quantity_text[:-1])
        # Handle Unicode fractions
        elif re.search(r"\d*\s*[\u00BC-\u00BE\u2150-\u215E]", quantity_text):
            quantity = unicodedata.numeric(quantity_text)
        # Handle English words
        else:
            quantity = w2n.word_to_num(quantity_text.lower())

        # Extract the remaining text as the unit
        unit_match = re.search(unit_regex, text[quantity_match.end(
        ):].strip() if quantity_match else text.strip())
        if unit_match:
            unit = unit_match.group(1).strip().lower()

    return quantity, unit

# Rate limit handling with exponential backoff


def handle_rate_limit(retries):
    wait_time = pow(2, retries)
    logging.info(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
    time.sleep(wait_time)


def generate_embedding(text):
    response = openai.Embedding.create(
        input=text, model="text-embedding-ada-002")
    embeddings = []
    for data_obj in response['data']:
        embeddings.append(data_obj['embedding'])
    return np.array(embeddings)


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
                    input=batch_texts.tolist(), model="text-embedding-ada-002")
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


def preprocess_data(df, df_file=None):
    if not df_file:
        # Extract Quantity and Unit from Text
        df[['Quantity', 'Unit']] = df['Text'].apply(
            lambda x: pd.Series(extract_quantity_and_unit(x)))
        
        # Generate embeddings for ingredient texts
        text_embeddings = generate_embeddings(df['Text'])
        ingredient_embeddings = generate_embeddings(df['Ingredient'])
        # Reshape embeddings to handle zero-dimensional arrays
        text_embeddings = text_embeddings.reshape(
            len(text_embeddings), -1) if text_embeddings.shape[0] > 0 else text_embeddings
        ingredient_embeddings = ingredient_embeddings.reshape(len(
            ingredient_embeddings), -1) if ingredient_embeddings.shape[0] > 0 else ingredient_embeddings

        df['Embeddings'] = text_embeddings + ingredient_embeddings

        # Fill missing Quantity with AmountNeeded
        df['Quantity'].fillna(df['AmountNeeded'], inplace=True)

        # Fill missing Unit with EACH
        df['Unit'].fillna('EACH', inplace=True)

        # One-hot encode the 'Unit' column
        unit_encoder = OneHotEncoder(sparse=False)
        unit_encoded = unit_encoder.fit_transform(df[['Unit']])
        unit_cols = [f"Unit_{unit}" for unit in unit_encoder.categories_[0]]
        df_encoded = pd.DataFrame(unit_encoded, columns=unit_cols)

        # Concatenate embeddings with the original DataFrame
        df = pd.concat(
            [df.drop(['Unit'], axis=1), df_encoded], axis=1)
        df.to_pickle("preprocessed.pkl")
    else:
        df = pd.read_pickle(df_file)

    return df


def train_model(dataset_file, df_file=None):
    # Load the dataset from JSON file
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)

    # Map dataset into DataFrame structure
    data = {
        'Text': [item['freeText'] for item in dataset],
        'Ingredient': [item['matchedIngredient']['name'] for item in dataset],
        'AmountNeeded': [item['amountNeededOfMatch'] for item in dataset]
    }
    df = pd.DataFrame(data)

    # Preprocess data and generate embeddings
    df = preprocess_data(df, df_file)

    # Split data into training and testing sets
    X = df.drop(['Ingredient', 'AmountNeeded'], axis=1)
    y_ingredient = df['Ingredient']
    y_amount = df['AmountNeeded']
    X_train, X_test, y_ingredient_train, y_ingredient_test, y_amount_train, y_amount_test = train_test_split(
        X, y_ingredient, y_amount, test_size=0.2, random_state=42)

    # Train XGBoost model for ingredient prediction
    model_ingredient = xgb.XGBRegressor(objective='reg:squarederror')
    model_ingredient.fit(X_train, y_ingredient_train)

    # Train XGBoost model for amount prediction
    model_amount = xgb.XGBRegressor(objective='reg:squarederror')
    model_amount.fit(X_train, y_amount_train)

    # Evaluate the models
    ingredient_r2_score = model_ingredient.score(X_test, y_ingredient_test)
    amount_r2_score = model_amount.score(X_test, y_amount_test)

    # Log the results
    logging.info("Ingredient Prediction R2 Score: %.2f", ingredient_r2_score)
    logging.info("Amount Prediction R2 Score: %.2f", amount_r2_score)

    # Save the trained models
    joblib.dump(model_ingredient, "model_ingredient.pkl")
    joblib.dump(model_amount, "model_amount.pkl")
    logging.info(
        "Model training complete. Saved as 'model_ingredient.pkl' and 'model_amount.pkl'.")


def predict_text(ingredient_model_file, amount_model_file, ingredient_text):
    # Load the trained models
    model_ingredient = joblib.load(ingredient_model_file)
    model_amount = joblib.load(amount_model_file)

    # Extract features from the input ingredient text
    quantity, unit = extract_quantity_and_unit(ingredient_text)

    # encoded units 
    encoded_unit_features = [x for x in model_ingredient.get_booster().feature_names if x.startswith("Unit_")]
    logging.debug(encoded_unit_features)
    # Convert the extracted unit to one-hot encoding
    unit_encoded = [0] * len(encoded_unit_features)
    if unit.lower() in [col_name[5:].lower() for col_name in encoded_unit_features]:
        unit_index = encoded_unit_features.index(f"Unit_{unit}")
        unit_encoded[unit_index] = 1

    text_embeddings = generate_embedding(ingredient_text)
    text_embeddings = text_embeddings.reshape(len(text_embeddings), -1)  # Reshape to 2-dimensional array

    # Prepare the input for prediction
    input_data = pd.DataFrame({
        'Text': [ingredient_text],
        'Embeddings': text_embeddings.tolist(),
        'Quantity': [quantity],
    })

    unit_encoded_df = pd.DataFrame([unit_encoded], columns=encoded_unit_features)
    df = pd.concat([input_data, unit_encoded_df], axis=1)

    logging.debug("input df is \n%s", df)
    # Predict the ingredient name
    ingredient_name = model_ingredient.predict(df)

    # Predict the amount needed
    amount_needed = model_amount.predict(df)[0]

    # Log the prediction results
    logging.info("Predicted Ingredient: %s", ingredient_name)
    logging.info("Predicted Amount Needed: %s", amount_needed)


@click.command()
@click.option('--train', is_flag=True, help='Train the model')
@click.option('--predict', is_flag=True, help='Predict using the trained model')
@click.option('--dataset-file', type=click.Path(exists=True), help='Path to the dataset file')
@click.option('--ingredient-model-file', type=click.Path(exists=True), help='Path to the ingredient model file')
@click.option('--amount-model-file', type=click.Path(exists=True), help='Path to the amount model file')
@click.option('--df-file', type=click.Path(exists=True), help='Optional path to preprocessed datafram file')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO', help='Logging level')
@click.argument('ingredient-text', default='', required=False)
def main(train, predict, dataset_file, ingredient_model_file, amount_model_file, df_file, log_level, ingredient_text):
    # Configure logger
    logging.basicConfig(level=log_level)
    if train:
        if not dataset_file:
            raise click.BadArgumentUsage(
                "Please provide the path to the dataset file using --dataset-file")
        train_model(dataset_file, df_file)
    elif predict:
        if not ingredient_model_file and not amount_model_file:
            raise click.BadArgumentUsage(
                "Please provide the path to the model file using --ingredient-model-file and --amount-model-file")
        if not ingredient_text:
            raise click.BadArgumentUsage(
                "Please provide the new text to predict using the model")
        predict_text(ingredient_model_file, amount_model_file, ingredient_text)
    else:
        raise click.BadArgumentUsage(
            "Please specify either --train or --predict")


if __name__ == '__main__':
    main()
