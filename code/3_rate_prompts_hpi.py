import pandas as pd
from code.utils import generate_ratings, read_all_files_to_dataframe

prompts = pd.read_parquet('input/journal_prompts.parquet')


# Prompt to code for each of the coaching practices
hpi_prompt = """
Here are six habits that lead to long-term success across domains, with example items to measure them.

DEFINITIONS:
Seek Clarity. High performers actively seek clarity on who they want to be, what they want to accomplish, and how they will achieve it. They are clear about their goals and passions. They have a clear vision of what they will achieve in life and how they will do it. High performers consistently seek clarity as times change and as they take on new projects or enter new social situations.
Generate Energy. High performers generate energy so that they can maintain consistent focus and effort throughout each day. They actively care for their bodies and minds to ensure that they can sustain high levels of energy over the long-term. This translates into greater physical energy, mental stamina, and positive emotions.
Raise Necessity. High performers experience a necessity for exceptional performance. They tap into the reasons why they absolutely must perform well, which produces a powerful drive to work hard and succeed. By combining both internal standards (e.g., identity, beliefs, values, or expectations for excellence) and external demands (e.g., social obligations, competition, public commitments, deadlines), they sustain a high level of motivation.
Increase Productivity. High performers spend their time working on the things that matter. This allows them to consistently produce outputs that truly count. They shield their attention from distractions and opportunities that would pull them away from what matters most. This allows them to stay productive day in and day out.
Develop Influence. High performers develop influence with the people around them. They learn how to get people to believe in and support their efforts and aspirations. By demonstrating strong leadership and being able to persuade people to contribute to important projects, they are able to make the major achievements that require a positive support network.
Demonstrate Courage. High performers demonstrate courage by expressing their ideas, taking bold action, and standing up for themselves and others. They do what they think is right even in the face of fear, uncertainty, threat, or changing conditions. Rather than viewing courage as an occasional act, it is treated as a consistent and deliberate choice.

EXAMPLE ITEMS:
Clarity Items
1. I know what I want - I'm clear about my goals and passions.
2. I have clarity about what I want to accomplish in my life.
3. I know exactly what will make me successful in the next three years of my life.
Energy Items
4. I have the energy needed to achieve my goals each day.
5. I feel highly energized every day.
6. I have the stamina needed to be present, enthusiastic, and focused throughout the day.
Necessity Items
7. I feel a deep emotional drive to succeed.
8. I feel a high level of motivation that consistently forces me to work hard, stay disciplined, and push myself.
9. I work exceptionally hard because I know there are many rewards to reap from success.
Productivity Items
10. I’m good at prioritizing and working on what’s important.
11. I’m good at being productive on the things that really count.
12. I’m consistently productive over the long-term.
Influence Items
13. I’m good at persuading people to do things.
14. I have strong leadership skills.
15. People in my network or life would describe me as highly influential.
Courage Items
16. I speak up for myself, even when it’s hard.
17. I respond quickly to life’s challenges and emergencies versus avoiding them.
18. I anticipate that new situations will involve difficulty or struggle and I’m comfortable with that.

I will give you a journaling prompt, and you will tell me which of the six indicators it is related to.

Structure your answer as a python dictionary with the indicator as the key and 0 or 1 for absence/presence in the value. If you are unsure, you can use decimals to quantify your uncertainty. Also, add a key for the estimated length that a thorough answer would take in words.
Example: {"clarity": 1, "energy": 0, "necessity": 0, "productivity": 0, "influence": 0, "courage": 0, "length": 100}
"""
prompts["prompt_id"] = prompts.index
prompts.columns

prompts_sample = prompts.sample(5)

import openai

test = generate_ratings(data = prompts_sample, id_col = "prompt_id", text_col = "text", prompt = hpi_prompt, output_dir = "output_hpi", verbose = True, temperature = 1, keep_details = True, format = "dictionary")

# Start of repeating block
import time
while lenpr > 0:
  pd.set_option('display.max_colwidth', 170)
  results_so_far,problematic_files = read_all_files_to_dataframe("output_hpi", format = "dictionary", keep_details = True)
  results_so_far
  problematic_files

  ids_so_far = results_so_far.id.unique()

  prompts_remaining = prompts[~prompts.prompt_id.isin(ids_so_far)]
  lenpr = len(prompts_remaining)
  prompts_sample = prompts_remaining.sample(9)
  test = generate_ratings(data = prompts_sample, id_col = "prompt_id", text_col = "text", prompt = hpi_prompt, output_dir = "output_hpi", verbose = True, temperature = 1, keep_details = True, format = "dictionary")
  time.sleep(1)
# End of repeating block

# pd.set_option('display.max_colwidth', None)
# read_all_files_to_dataframe("output_journal", format = "list").to_latex("output_journal.tex", index = False, longtable = True)

## Put it all together
results_so_far = results_so_far.sort_values("id")
# rename id as prompt_id
results_so_far = results_so_far.rename(columns = {"id": "prompt_id"})

# merge results with the original data
results = prompts.merge(results_so_far, on = "prompt_id")
# save to parquet
results.to_parquet("input/journal_prompts_rated.parquet", index = False)
results.to_csv("output/journal_prompts_rated.csv", index = False)