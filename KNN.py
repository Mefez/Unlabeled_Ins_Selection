import multiprocessing as mp
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt
import numpy as np
import time
import math
import json

epsilon=0.01
minimum_samples=5 #3

#0.0001,2 04:13:43 clustering time

class PE:
  def __init__(self, positions, sha, label, ne, NP):
    self.psitions = positions #np.mod(positions[:2],[200,200]) #n-ary position
    self.label = label
    self.sha = sha
    self.NE = ne
    self.NP = NP

  def __str__(self):
    return f"{{\"sha256\": \"{self.sha}\", \"histogram\": {self.psitions}, \"label\": {self.label}, \"NE\": \"{self.NE}\", \"NP\": {self.NP}}}"
  
def KNN(Obj, listah):
  ret=[]
  test_result = 0
  flag = 0

  for i in range(0, minimum_samples):
    if Obj.sha == listah[i].sha:
      flag = flag + 1
  
  for i in range(0, minimum_samples+flag):
    if (Obj.sha != listah[i].sha) or (listah[i].label != -1):
      ret.append(listah[i])
    
  for i in range(minimum_samples+flag, len(listah)):
    for j in range(0, minimum_samples):
      if (math.dist(Obj.psitions, listah[i].psitions) < math.dist(Obj.psitions, ret[j].psitions)) and (Obj.sha != listah[i].sha) and (listah[i].label != -1):
        ret[j] = listah[i]

  for i in range(0, minimum_samples):
    test_result = test_result + ret[i].label
  
  if (test_result >= 3 and Obj.label == 1) or (test_result < 3 and Obj.label == 0):
    Obj.label = 1
    with open("third_step_result.jsonl", "a") as f:
      f.write(str(Obj) + "\n")
    f.close()
  else:
    Obj.label = 0
    with open("third_step_result.jsonl", "a") as f:
      f.write(str(Obj) + "\n")
    f.close()


def get_result(result):
    global results
    results.append(result)


if __name__ == '__main__':
  strlist = []
  results = []
  
  """with open("final_result_0.001,3.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    strlist.append(json_str['sha256'])
  str_listah = np.array(strlist)

  mylist = []

  with open("/scratch/zorlumeh/ember/train_features_0.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['histogram'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_1.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['sha256'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_2.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['histogram'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_3.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['histogram'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_4.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['histogram'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_5.jsonl", "r") as json_file:
    
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    if json_str['histogram'] in str_listah:
        mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))



  result_listah = np.array(mylist)

  plotdata = []
  
  for i in result_listah:
    plotdata.append(i.psitions)
  
  numOfFeatures = 50
  pca = PCA(n_components=numOfFeatures)
  model = pca.fit(plotdata)
  X_pca = np.array(model.transform(plotdata)).tolist()
  sc = StandardScaler()
  plotdata = np.array(sc.fit_transform(X_pca)).tolist()

  with open("first_step_result.jsonl", "a") as f:
    for i, j in zip(result_listah, plotdata):
      i.psitions = j
      f.write(str(i) + "\n")
  f.close()

  #################################################################################


  mylist = []

  with open("/scratch/zorlumeh/ember/test_features.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], -1, -1))
  test_listah = np.array(mylist)

  plotdata = []
  
  for i in test_listah:
    plotdata.append(i.psitions)
  
  numOfFeatures = 50
  pca = PCA(n_components=numOfFeatures)
  model = pca.fit(plotdata)
  X_pca = np.array(model.transform(plotdata)).tolist()
  sc = StandardScaler()
  plotdata = np.array(sc.fit_transform(X_pca)).tolist()

  with open("second_step_result.jsonl", "a") as f:
    for i, j in zip(test_listah, plotdata):
      i.psitions = j
      f.write(str(i) + "\n")
  f.close()"""

  #################################################################################

  """data = []
  result = []

  with open("first_step_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  result_listah = np.array(result)

  with open("second_step_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  test_listah = np.array(result)

  pool = mp.Pool(mp.cpu_count())

  for i in range(0, len(test_listah)):
    pool.apply_async(KNN, args=(test_listah[i], result_listah), callback=get_result)
  
  pool.close()
  pool.join()"""

  #################################################################################
  correct = 0
  data = []
  result = []

  with open("third_step_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  result_listah = np.array(result)

  for i in result_listah:
    if i.label == 1:
      correct = correct + 1
  
  with open("fourth_step_result.jsonl", "a") as f:
    f.write(str(correct) + "\n" + str(len(result_listah)))
  f.close()