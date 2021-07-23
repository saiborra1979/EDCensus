# Script to process HeroAI data extracts
import os
import numpy as np
import pandas as pd
from funs_support import find_dir_olu

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_rt = os.path.join(dir_olu, 'rt')

#############################
# --- STEP 1: LOAD DATA --- #

# Existing output data
df_demo = pd.read_csv(os.path.join(dir_flow,'demo4flow.csv'))

qq = ['disposition_selected', 'bed_requested', 'bed_ready', 'Length_of_stay']

# JSON real-time data
fn_rt = 'raw_head.json'
df_json = pd.read_json(os.path.join(dir_rt,fn_rt),lines=True)

##############################
# --- STEP 2: PARSE JSON --- #

df_json.loc[0]





