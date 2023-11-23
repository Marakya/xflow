from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

df = pd.read_csv('/home/xflow/project/datasets/data_test.csv', header=None)
df.columns = ['id', 'counts']
df['counts'] = df['counts'].fillna(0)

model = RandomForestRegressor(max_depth=2, random_state=0)
with open('/home/xflow/project/models/data.pickle', 'rb') as f:
    model = pickle.load(f)

score = model.score(df['id'].values.reshape(-1,1), df['counts'])
print("score=", score)
