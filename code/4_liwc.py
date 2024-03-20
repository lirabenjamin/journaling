from code.utils import process_dataframe_with_liwc
import pandas as pd

# Load the data
data = pd.read_parquet('input/journal_prompts_rated.parquet')

data = data[["prompt_id", "content"]]

# Process the data with LIWC
liwc_data = process_dataframe_with_liwc(data, id_column="prompt_id", text_column="content")
liwc_data.to_parquet('input/liwc_output.parquet', index=False)
liwc_data.columns.tolist()

liwc_data = liwc_data[["Row ID", "focuspast","focuspresent","focusfuture"]]

#rename rowid as prompt_id
liwc_data = liwc_data.rename(columns={'Row ID':'prompt_id'})

# softmax the focuspast, focuspresent, focusfuture
import numpy as np
def softmax(x):
    # Subtract the max for numerical stability (reshaping to keep correct dimensions)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Divide by the sum to get probabilities (again, keepdims for correct broadcasting)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Applying softmax
softmax_results = softmax(liwc_data[["focuspast", "focuspresent", "focusfuture"]].values)

# Add the softmax results to the dataframe
liwc_data["focuspast_softmax"] = softmax_results[:,0]
liwc_data["focuspresent_softmax"] = softmax_results[:,1]
liwc_data["focusfuture_softmax"] = softmax_results[:,2]

# merge to data
data = pd.read_parquet('input/journal_prompts_rated.parquet')
data.columns
data = data[["id", "text", "prompt_id", "content",'clarity', 'energy', 'necessity','productivity', 'influence', 'courage', 'length']]

# rename
data.rename(columns={'content':'prompt', 'text':'rapid_fire'}, inplace=True)

data = data.merge(liwc_data, on="prompt_id")

data.to_parquet('output/journal_prompts_rated_with_liwc.parquet', index=False)
data.to_csv('output/journal_prompts_rated_with_liwc.csv', index=False)