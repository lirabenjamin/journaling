import pandas as pd
import os
import openai
import concurrent.futures
import datetime
import ast
    

def read_all_files_to_dataframe(directory, format = "dictionary", keep_details = True):
  output_dir = directory
  all_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]
  df_list = []
  
  problematic_files = []
  
  for filename in all_files:
        with open(filename, 'r') as f:
            content = f.read()
            try:
                # Attempt to evaluate the content
                evaluated_content = ast.literal_eval(content)
                df_list.append({"filename": filename, "content": evaluated_content})
            except SyntaxError:
                # Log problematic file and skip
                print(f"Skipping problematic file due to SyntaxError: {filename}")
                problematic_files.append(filename)
                continue
              
  df =  pd.DataFrame(df_list)
  if df.empty:
      raise ValueError("No data read from the output directory. Ensure .txt files exist in the directory.")
  id_col = "id"
  df.columns = [id_col, "content"]
  df[id_col] = df[id_col].str.replace(f"{output_dir}/", "")
  df[id_col] = df[id_col].str.replace(".txt", "")
  df[[id_col, "timestamp", "temperature"]] = df[id_col].str.split("_", expand=True)

  # Unroll the dictionary
  if format == "dictionary":
    dict_data = pd.DataFrame(df.content.tolist()) 
    # Combine df and combined_df
    df = pd.concat([df.drop("content", axis=1), dict_data], axis=1)
    
    
    
  if format == "list":
    df = df.explode('content')
  if not keep_details:
      df = df.drop(["content", "timestamp", "temp"], axis=1)
  
  # Join df with data on id
  df[id_col] = df[id_col].astype(int)

  return df,problematic_files


def generate_ratings(data: pd.DataFrame, id_col: str, text_col: str, prompt: str, output_dir: str, verbose: bool = False, temperature = 1, keep_details = True, format = "dictionary") -> pd.DataFrame:
    
    # Check OpenAI API key
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please set it before calling this function.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def rate_conversation(id, conversation):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Here is the journaling prompt:\n{conversation}"},
            ]
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f"{output_dir}/{id}_{now}_temp1.txt", "w") as f:
            f.write(result)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(rate_conversation, data[id_col], data[text_col])
    
    
    df,problematic_files = read_all_files_to_dataframe(output_dir, format = format, keep_details = keep_details)
    return df,problematic_files
    

def process_dataframe_with_liwc(df: pd.DataFrame, id_column:str, text_column: str, save_to_csv: bool = False, output_filename: str = 'liwc_output.csv') -> pd.DataFrame:
    import pandas as pd
    import subprocess
    import os
    # First, save the DataFrame to a temporary CSV file
    temp_filename = 'temp_for_liwc.csv'
    df.to_csv(temp_filename, index=False)

    # Command to run LIWC on the CSV file
    cmd_to_execute = ["LIWC-22-cli",
                      "--mode", "wc",
                      "--input", temp_filename,
                      "--row-id-indices", str(df.columns.get_loc(id_column)+1),  # Assuming the first column (index 0) is the identifier
                      "--column-indices", str(df.columns.get_loc(text_column)+1),  # Index of the text column
                      "--output", output_filename if save_to_csv else 'liwc_temp_output.csv']

    # Execute the command
    result = subprocess.call(cmd_to_execute)

    # Check if the command was successful
    if result != 0:
        os.remove(temp_filename)
        raise RuntimeError("Error occurred while running LIWC-22. Ensure the LIWC-22 application is running.")

    # Read the LIWC output into a pandas DataFrame
    if os.path.exists(output_filename if save_to_csv else 'liwc_temp_output.csv'):
        liwc_output = pd.read_csv(output_filename if save_to_csv else 'liwc_temp_output.csv')
    else:
        os.remove(temp_filename)
        raise FileNotFoundError(f"Expected output file {output_filename if save_to_csv else 'liwc_temp_output.csv'} not found.")
    
    # Clean up temporary files
    os.remove(temp_filename)
    if not save_to_csv:
        os.remove('liwc_temp_output.csv')

    return liwc_output

import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
