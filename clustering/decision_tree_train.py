import os
import json
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys
import pickle


def dbscan_prediction(data, label, input_vector, thres):
    npdata = np.array(data)
    npinput = np.array(input_vector)
    distance = np.linalg.norm(npdata - npinput, axis=1)
    if min(distance) < thres:
        index = np.argmin(distance)
        return label[index]
    else:
        return -1


def decision_tree_table(label, position, imgs):
    d = max(label)+2
    n = len(imgs)
    x = np.zeros((n, d))
    for c in range(len(label)):
        i, _, _ = position[c]
        j = label[c]
        x[i, j] = 1
    return x


def decision_tree_dbscan(label, position, imgs):
    unique_path = list(dict.fromkeys(imgs))
    y = [ img.split("/")[-2] for img in unique_path ]
    x = decision_tree_table(label, position, unique_path)
    clf = DecisionTreeClassifier(random_state=0).fit(x, y)
    return clf


def dbscan_save(data, label, position, imgs):
    n = len(label)
    path = [None] * n
    c = 0
    for item in position:
        path[c] = imgs[item[0]][0]
        c = c + 1
    result = {"label": label.tolist(), "data": data.tolist(), "position":position, "path": path}
    return result


def dbscan_load(path):
    result = json.load(open(path, "rb"))
    return result


def save_model(model, path):
    sys.setrecursionlimit(50000)
    pickle.dump(model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

model_name = "dbscan_core_4_0.6.json"
model_path = "/Users/dylan/Twente/Capita Selecta/project/clustering/model"
model = dbscan_load(os.path.join(model_path, model_name))
##### DECISION TREE CLASSIFIER
embedded_vector = model["data"]
label = model["label"]
position = model["position"]
image_paths = model["path"]
print("Start Decision Tree Classifier...", flush=True)
decision_tree_model = decision_tree_dbscan(label, position, image_paths)
model_name = "decision_tree_0.6.pkl"
### SAVE DECISION TREE MODEL
path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/")
if not os.path.exists(path):
    os.makedirs(path)
save_model(decision_tree_model, os.path.join(path,model_name))


