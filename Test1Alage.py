# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#for dirname, _, filenames in os.walk('/kaggle/input/research-on-algae-growth-in-the-laboratory/algeas.csv'):
  #  for filename in filenames:
 #       print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
##
file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"
df = pd.read_csv(file_path)
df