import requests
import pandas as pd
import json
from pyyoutube import Api
import os
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

key = "AIzaSyCgMr4McKwt4YEj_-wR2h5GVC_90pQT2qk"
api = Api(api_key=key)

query = "'Machine Learning'"
video = api.search_by_keywords(q=query, search_type=["video"], count=100, limit=300)

maxResults = 100000
nextPageToken = ""
s = 0

with mlflow.start_run():
    dic={}
    for id_ in [x.id.videoId for x in video.items]:
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
                "key={}&textFormat=plainText&" + \
                "part=snippet&" + \
                "videoId={}&" + \
                "maxResults={}&" + \
                "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        c=0
        for item in data['items']:
            c+=1
            if item['snippet']['topLevelComment']['snippet']['channelId'] in dic.keys():
                dic[item['snippet']['topLevelComment']['snippet']['channelId']] += item['snippet']['topLevelComment']['snippet']['likeCount']
            else:
                dic[item['snippet']['topLevelComment']['snippet']['channelId']] = item['snippet']['topLevelComment']['snippet']['likeCount']

    df = pd.DataFrame.from_dict(dic, orient='index')
    df = df.reset_index()
    df.rename(columns = {0:'counts', 'index':'id'}, inplace = True )

    mlflow.log_artifact(local_path="/home/xflow/project/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

df.to_csv('/home/xflow/project/datasets/data.csv')