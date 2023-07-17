import json
import logging
import re
import unicodedata
from fractions import Fraction

from word2number import w2n

logging.basicConfig(level='INFO', filemode='w', filename='logs.txt')


def extract_fraction(string):
    # Remove any leading/trailing whitespace
    string = string.strip()

    # Regular expression pattern to extract fractional part
    pattern = r"\d+/\d+"

    # Find the first occurrence of the fractional part
    match = re.search(pattern, string)

    if match:
        # Extract the fractional part from the matched substring
        fraction_string = match.group()

        # Convert the fractional part to a float
        try:
            fraction = Fraction(fraction_string)
            fraction_float = float(fraction)
            return fraction_float, match
        except ValueError:
            pass

    # If no fraction is found or if the conversion fails, return None
    return None, None


def extract_ingredient_details(text):
    # Define common units used in recipe ingredients
    common_units = [
        '(?:\d+(\.\d+)?-ounce )jars', '(?:\d+(\.\d+)?-ounce )jar',
        '(?:\d+(\.\d+)?-ounce )cans', '(?:\d+(\.\d+)?-ounce )can',
        'cloves', 'clove',
        'heads', 'head',
        'cups', 'cup', 'teaspoons', 'teaspoon', 'tablespoons', 'tablespoon',
        'ounces', 'ounce', 'pounds', 'pound', 'milligrams', 'milligram',
        'kilograms', 'kilogram', 'grams', 'gram',
        'mls', 'ml', 'milliliters', 'milliliter', 'liters', 'liter', 'gallons', 'gallon',
        'pinches', 'pinch', 'dashes', 'dash', 'slices', 'slice', 'cloves', 'clove',
        'sprigs', 'sprig', 'bunches', 'bunch',
    ]
    # Regular expressions to extract quantity and unit
    quantity_regex = r"((\d+(\.\d+)?)|\d*\s*[\u00BC-\u00BE\u2150-\u215E])"
    unit_regex = r'({})'.format('|'.join(common_units))

    # Extract quantity
    quantity = None
    unit = None

    # Extract quantity and unit from the first match
    quantity_match = re.search(quantity_regex, text.lower())
    fraction_float, fraction_match = extract_fraction(text)
    final_match = None
    ingredient_start = None
    if fraction_match:
        quantity = fraction_float
        final_match = fraction_match
    elif quantity_match:
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

        final_match = quantity_match

    ingredient_start = final_match.end() if final_match else None
    if ingredient_start:
        logging.debug(
            f"matched quantity {quantity} from {text} and will trim to {text[ingredient_start:]}")
    else:
        logging.debug(f"quantity is {quantity} from {text}")
    # Extract the remaining text as the unit
    unit_search_text = text[ingredient_start:] if ingredient_start else text
    logging.debug(f"searching {unit_search_text}")
    unit_match = re.search(
        unit_regex, unit_search_text, re.IGNORECASE)
    if unit_match:
        logging.debug(
            f"unit matches are {unit_match.groups()} in {unit_search_text}")
        unit = unit_match.group(1).strip().lower()
        ingredient_start = unit_match.end(
        ) + ingredient_start if ingredient_start else unit_match.end()

    if quantity is None:
        quantity = 1

    if unit is None:
        unit = 'EACH'

    ingredient_text = text[ingredient_start:].strip(
    ) if ingredient_start else text.strip()
    if not ingredient_text:
        ingredient_text = text.strip()
    logging.debug(f"{text} -> [{quantity}, {unit}, {ingredient_text}]")

    return ingredient_text, quantity, unit


def process_row(row):
    free_text = row["freeText"]
    ingredient_text, ingredient_quantity, ingredient_unit = extract_ingredient_details(
        free_text)
    target_quantity = row["amountNeededOfMatch"]
    target_ingredient = row["matchedIngredient"]["name"]
    target_unit = row["matchedIngredient"]["primaryUnit"]

    output_object = {
        "ingredient_text": ingredient_text,
        "ingredient_quantity": ingredient_quantity,
        "ingredient_unit": ingredient_unit,
        "target_ingredient": target_ingredient,
        "target_quantity": target_quantity,
        "target_unit": target_unit
    }
    logging.info(f"OUTPUT for {row} -> {output_object}")
    return output_object


# Load the input JSON file into a Python object
with open("./input_data.json", "r") as file:
    input_data = json.load(file)

# Process each row and generate the output objects
output_data = []
for row in input_data:
    output_row = process_row(row)
    output_data.append(output_row)

# Write the output JSON file
with open("./output_data.json", "w") as file:
    json.dump(output_data, file, indent=4)
