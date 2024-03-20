import os
import pandas as pd

data = pd.read_table('input/data.txt', delimiter="XXXXXX", header=None,  names = ['name'])

data['title'] = data['name'].str.startswith("Daily Fire")

# get the first string of the name
data['starts_num'] = data['name'].str[0].str.isdigit()

# tell me when name starts with a numbers
data.query('starts_num == True | title == True').to_csv('input/titles.csv', index=False)

data["title2"] = data['title'] | data['starts_num']

# Add blocks together
# Process to concatenate text
text_blocks = []
current_block = ""

for index, row in data.iterrows():
    if row['title2']:  # If current row is a start of a new block
        if current_block:  # If there's a current block, save it before starting a new one
            text_blocks.append(current_block)
            current_block = ""  # Reset the current block
        current_block = row['name']  # Start new block with current row's text
    else:  # If not a new block, append current row's text to the existing block
        current_block += " " + row['name']

# Don't forget to add the last block if it exists
if current_block:
    text_blocks.append(current_block)

text_blocks = pd.DataFrame(text_blocks, columns=['text'])




# Prompt to code for past, present, and future
prompt_length = """
I will give you a journaling prompt, and you will answer with the probability that it is related to the past, present, or future.

Structure your answer as a python dictionary with past/present/future as the key and 0/1 as the values to indicate whether it has that time focus or not. If you are unsure, you can use decimals to quantify your uncertainty.
Example: {"past": 0, "present": 0, "future": 1}
"""


# Prompt how much writing would this be.
prompt_length = """
I will give you a journaling prompt, and you will tell me how many words you think it would take to answer it thoroughougly.

Structure your answer as a python dictionary with "length" as the key and an average number of words as the value.
"""


text_blocks['id'] = text_blocks.index

text_blocks.to_parquet('input/text_blocks.parquet', index=False)
