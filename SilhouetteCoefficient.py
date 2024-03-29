import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import json

class PE:
  def __init__(self, positions, sha, label, ne, NP):
    self.psitions = positions
    self.label = label
    self.sha = sha
    self.NE = ne
    self.NP = NP

  def __str__(self):
    return f"{{\"sha256\": \"{self.sha}\", \"histogram\": {self.psitions}, \"label\": {self.label}, \"NE\": \"{self.NE}\", \"NP\": {self.NP}}}"


result = []

with open("final_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
json_file.close

for json_str in data:
    result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
mistah_listah = np.array(result)

plotdata = []
labels = [] 

for i in range(0, len(mistah_listah)):
   labels.append(mistah_listah[i].label)
   plotdata.append(mistah_listah[i].psitions)

print(plotdata[0])

score = metrics.silhouette_score(plotdata, labels, metric='euclidean')

print(score)