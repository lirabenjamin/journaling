import pandas as pd
from code.utils import generate_ratings, read_all_files_to_dataframe

text_blocks = pd.read_parquet('input/text_blocks.parquet')

# Prompt to make into a standalone journaling prompt

prompt_journal = """
You are an expert in coaching. You have been asked to write a journaling prompt for a coaching practice.

As input, here are some notes. They may contain a title, some reflection framework, and some reflection questions.

Your task is to write a journaling prompt that can be used for a coaching practice. It should be COMPLETELY self contained. It should be clear and concise. It should be easy to understand and follow. If you are introducing terminology make sure to explain it first.

If needed, the prompt should provide start with a sentence or two to provide context, introduce the topic, provide a framework, or set the stage for the reflection. It should then be followed by a question or a set of questions.

It should take the user through a process of reflection. It should be open ended and allow for a variety of responses. It should be thought provoking and insightful.

The user should take about 15 minutes to complete the exercise. If there is too much material, break it down into digestible pieces that can be completed in 15 minutes.

I will tell you the input, and expect you to write the journaling prompt.

Do not add a date to it, as it should be timeless. It should be able to be used at any time.
Structure your answer as a python list with each entry being a self contained short jouranling prompt. 

Every item in the list should be independent of each other. Do NOT start the prompts with "Next", "Lastly" or "Finally" or anything indicating sequence, because users will see them out of context.

Do not add any text other than the list of prompts. DO NOT START YOUR ANSWER WITH "HERE ARE YOUR PROMPTS" OR ANYTHING SIMILAR. JUST THE PROMPTS. DO NOT TITLE THE COLLECTION.
Make sure to close all quotation marks appropriately so python can parse it correctly.
Example: ["Prompt1" , "Prompt2", "Prompt3", "Promptn"]
"""

text_blocks_sample = text_blocks.sample(5)

test = generate_ratings(data = text_blocks_sample, id_col = "id", text_col = "text", prompt = prompt_journal, output_dir = "output_journal", verbose = True, temperature = 1, keep_details = True, format = "list")

# Start of repeating block
pd.set_option('display.max_colwidth', 170)
results_so_far,problematic_files = read_all_files_to_dataframe("output_journal", format = "list", keep_details = True)
results_so_far
problematic_files

ids_so_far = results_so_far.id.unique()

text_blocks_remaining = text_blocks[~text_blocks.id.isin(ids_so_far)]
text_blocks_sample = text_blocks_remaining.sample(15)
test = generate_ratings(data = text_blocks_remaining, id_col = "id", text_col = "text", prompt = prompt_journal, output_dir = "output_journal", verbose = True, temperature = 1, keep_details = True, format = "list")
# End of repeating block

# pd.set_option('display.max_colwidth', None)
# read_all_files_to_dataframe("output_journal", format = "list").to_latex("output_journal.tex", index = False, longtable = True)

## Put it all together
results_so_far = results_so_far.sort_values("id")
# merge results with the original data
results = text_blocks.merge(results_so_far, on = "id")
# save to parquet
results.to_parquet("input/journal_prompts.parquet", index = False)