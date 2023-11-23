import pandas as pd

df = pd.read_csv('/home/xflow/project/datasets/data.csv', header=None)

for i in range(len(df['id'])):
  max_= df['counts'].max()
  min_ =df['counts'].min()
  df['counts'][i] = (df['counts'][i]-min_)/(max_-min_)

df.to_csv('/home/xflow/project/datasets/data_processed.csv')