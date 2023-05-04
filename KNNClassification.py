import multiprocessing as mp
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import math
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


def get_result(result):
    global results
    results.append(result)


if __name__ == '__main__':
  strlist = []
  results = []
  
  with open("final_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    strlist.append(json_str['sha256'])
  str_listah = np.array(strlist)

  mylist = []

  with open("train_features_1.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("train_features_2.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("train_features_3.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("train_features_4.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("train_features_5.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))


  result_listah = np.array(mylist)

  with open("first_step_result.jsonl", "a") as f:
    for i in result_listah:
      f.write(str(i) + "\n")
  f.close()

  #################################################################################


  mylist = []

  with open("/home/zorlumeh/ember/test_features.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))
  test_listah = np.array(mylist)

  with open("second_step_result.jsonl", "a") as f:
    for i in test_listah:
      f.write(str(i) + "\n")
  f.close()

  #################################################################################

  data = []
  result = []

  with open("first_step_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  result_listah = np.array(result)

  result = []

  with open("second_step_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  test_listah = np.array(result)

  train_data = []
  train_labels = []

  for i in result_listah:
    train_data.append(i.psitions)
    train_labels.append(i.label)
    
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(train_data, train_labels) 

  test_data = []
  test_labels = []

  for i in test_listah:
    test_data.append(i.psitions)
    test_labels.append(i.label)

  predicted = knn.predict(test_data)

  ct = 0

  for i in range(0, len(test_listah)):
    if predicted[i] == test_labels[i]:
        ct = ct + 1

  with open("knn_result.jsonl", "a") as f:
    f.write("\n" + str(ct) + "\n" + str(len(test_listah)))
  f.close()