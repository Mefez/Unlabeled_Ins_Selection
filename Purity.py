import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import json

class PE:
  def __init__(self, positions, sha, label, cluster):
    self.psitions = positions
    self.label = label
    self.sha = sha
    self.cluster = cluster

  def __str__(self):
    return f"{{\"sha256\": \"{self.sha}\", \"histogram\": {self.psitions}, \"label\": {self.label}, \"cluster\": \"{self.cluster}}}"


result = []

with open("final_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
json_file.close

for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], -1, json_str['label']))
  
mistah_listah = np.array(result)

###################################################################################################################3
print(len(mistah_listah))
result = {}

with open("train_features_1.jsonl", "r") as json_file:
  data = [json.loads(line) for line in json_file]
json_file.close

for json_str in data:
    result[json_str['sha256']] = json_str['label']

finalist = []

for i in range(0, len(mistah_listah)):
    finalist.append([])

for i in range(0, len(mistah_listah)):
    mistah_listah[i].label = result[mistah_listah[i].sha]
    finalist[mistah_listah[i].cluster].append(i)
    

total = 0
ct = 0

for i in range(0, len(mistah_listah)):
    m = 0
    b = 0

    if len(finalist[i]) != 0:
      for j in range(0, len(finalist[i])):
        if mistah_listah[finalist[i][j]].label == 0:
           b = b+1
        elif mistah_listah[finalist[i][j]].label == 1:
           m = m+1
        elif mistah_listah[finalist[i][j]].label == -1:
           ct = ct +1 
    
    if b > m:
        total = total + b
    else:
        total = total + m

print(total/len(mistah_listah))