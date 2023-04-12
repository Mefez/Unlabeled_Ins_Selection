import multiprocessing as mp
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt
import numpy as np
import time
import math
import json

epsilon=0.001
minimum_samples=3

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
  
def get_result(result):
    global results
    results.append(result)

def NE(Obj, listah):
  dist = 0

  for i in range(0, len(listah)):
    if Obj.label != listah[i].label:
      Obj.NE = listah[i].sha
      Obj.NP = i
      dist = math.dist(Obj.psitions, listah[i].psitions)
      break

  for i in range(0, len(listah)):
    if Obj.label != listah[i].label and dist > math.dist(Obj.psitions, listah[i].psitions):
      dist = math.dist(Obj.psitions, listah[i].psitions)
      Obj.NE = listah[i].sha
      Obj.NP = i

  with open("NE_result.jsonl", "a") as f:
    f.write(str(Obj) + "\n")
  f.close()

  return Obj

def NCE(Obj, listah, finalist):

  counter = 0

  dist = math.dist(Obj.psitions, listah[Obj.NP].psitions)

  for i in finalist:
    if dist > math.dist(listah[i].psitions, listah[Obj.NP].psitions):
      counter += 1
    
    if counter == minimum_samples:
      break

  if counter < minimum_samples:
    with open("final_result.jsonl", "a") as f:
      f.write(str(Obj) + "\n")
    f.close()

if __name__ == '__main__':
  mylist = []
  results = []
  
  with open("/scratch/zorlumeh/ember/train_features_0.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))
  
  with open("/scratch/zorlumeh/ember/train_features_1.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_2.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_3.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_4.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))

  with open("/scratch/zorlumeh/ember/train_features_5.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close
  for json_str in data:
    mylist.append(PE(json_str['histogram'], json_str['sha256'], -1, -1, -1))

  mistah_listah = np.array(mylist)

  #################################################################################

  plotdata = []
  
  for i in mistah_listah:
    plotdata.append(i.psitions)

  #for i in range(0, 2):
  #  print (plotdata[i])
  
  numOfFeatures = 50 #[set here number of features, e.g. 50]
  pca = PCA(n_components=numOfFeatures)
  model = pca.fit(plotdata) #X is your dataset (it is better to normalized it before PCA)
  X_pca = np.array(model.transform(plotdata)).tolist()
  sc = StandardScaler()
  plotdata = np.array(sc.fit_transform(X_pca)).tolist()
  

  #for i in range(0, 2):
  #  print (X_pca[i])

  #for i in range(0, 2):
  #  print (plotdata[i])

  #plotdata = np.array(plotdata).reshape(len(mistah_listah),2)

  clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(plotdata)

  for i, j in zip(mistah_listah, clustering.labels_):
    i.label = j

  with open("clustering_result.jsonl", "a") as f:
    for i, j in zip(mistah_listah, plotdata):
      i.psitions = j #np.mod(j['histogram'][:2],[200,200]).tolist()
      f.write(str(i) + "\n")
  f.close()

  #################################################################################

  data = []
  result = []

  with open("clustering_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    if json_str['label'] != -1:
      result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  mistah_listah = np.array(result)

  pool = mp.Pool(mp.cpu_count())

  for i in range(0, len(mistah_listah)):
    pool.apply_async(NE, args=(mistah_listah[i], mistah_listah), callback=get_result)
  
  pool.close()
  pool.join()

  #################################################################################

  data = []
  result = []

  with open("NE_result.jsonl", "r") as json_file:
    data = [json.loads(line) for line in json_file]
  json_file.close

  for json_str in data:
    result.append(PE(json_str['histogram'], json_str['sha256'], json_str['label'], json_str['NE'], json_str['NP']))
  
  mistah_listah = np.array(result)

  finalist = []
  for i in range(0, len(mistah_listah)):
    finalist.append([])

  for i in range(0, len(mistah_listah)):
    finalist[mistah_listah[i].label].append(i)

  pool = mp.Pool(mp.cpu_count())

  for i in range(0, len(mistah_listah)):
    pool.apply_async(NCE, args=(mistah_listah[i], mistah_listah, finalist[mistah_listah[i].label]), callback=get_result)
  
  pool.close()
  pool.join()

  """finalist = np.zeros(len(mistah_listah))

  #print (len(mistah_listah))
  
  for i in mistah_listah:
    finalist[i.NP] = 1
  
  for i in range(0, len(mistah_listah)):
    if finalist[i] == 0:
      mistah_listah[i].label = -1

  with open("final_result.jsonl", "a") as f:
    for i in mistah_listah:
      if i.label != -1:
        f.write(str(i) + "\n")
  f.close()"""