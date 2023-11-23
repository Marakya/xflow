import pandas as pd
import numpy as np

df = pd.read_csv('/home/xflow/project/datasets/data_processed.csv', header=None)

l = int(len(df)*0.7)
train = df[:l]
test = df[l+1:]

train.to_csv('/home/xflow/project/datasets/data_train.csv', 
                        header=None, 
                        index=None)
test.to_csv('/home/xflow/project/datasets/data_test.csv', 
                        header=None, 
                        index=None)
