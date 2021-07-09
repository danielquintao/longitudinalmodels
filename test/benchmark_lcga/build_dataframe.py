# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

def build_dataframe(y, dfname, time, degree):
    data = pd.DataFrame(y, columns=time)
    data['individual'] = data.index
    data = pd.melt(data, id_vars='individual', value_vars=time, var_name='time', value_name='observation')
    data['time'] = pd.to_numeric(data['time'])
    data.to_csv(dfname+'_to_df.csv')