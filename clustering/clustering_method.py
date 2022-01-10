from torch.utils.data import DataLoader
import numpy as np
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pickle
import json
import sys
import time
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift, DBSCAN, OPTICS, Birch
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# import faiss

# 1. get output from bagnet 128-D vector (patch - cluster by patch)
# 2. feed the vector to clustering method
# 3. Get the result
# 4. trace back to the input, grouping them together

####### CLUSTER METHOD
### 0. Affinity propagation

def clustering(model, input_channel, dataLoader: DataLoader, foldername: str, device, args: argparse.Namespace, clusterMethod: int, training: bool, model_path=""):
    modelname = args.net

    if 'bagnet' not in modelname: #modelname can be "bagnet33" leading to patchsize = 33
        raise ValueError
    else:
        patchsize = modelname[-2:]
        if not patchsize.isdigit(): #if receptivefield < 10
            patchsize = modelname[-1:]

    patchsize = int(patchsize)

    model.eval()

    dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    ###### set up images
    imgs = dataLoader.dataset.imgs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_normalize = transforms.Compose([
                            transforms.Resize(size=(224,224)),
                            transforms.ToTensor(),
                            normalize
                        ])


    ####### init clustered input
    skip_const = 4
    D1 = len(imgs)
    D2 = int(24 / skip_const)
    D3 = int(24 / skip_const)
    reshaped_img_enc = np.empty([D1*D2*D3, input_channel])
    reshaped_img_enc_pos = [None] * (D1*D2*D3)
    c = 0

    
    #### skipping patches
    for i, image in enumerate(imgs):
        img = Image.open(image[0])
        if len(img.getbands()) != 3:
            img = img.convert(mode='RGB')
        img_normalized_tensor = transform_normalize(img).unsqueeze_(0).to(device)

        with torch.no_grad():
            # l2_norm_img
            img_enc = model.forward(img_normalized_tensor)
            # [64, 128, 24, 24]
            # [64, 1, 24, 24]
        img_enc = img_enc.squeeze(0)
        img_shape = img_enc.shape
        img_enc = img_enc.to(torch.device('cpu'))
        

        # img_id = i*24*24 + j*24 + k
        for j in range(img_shape[1]):
            if j % skip_const == 0:
                for k in range(img_shape[2]):
                    if k % skip_const == 0:
                        reshaped_img_enc[c] = img_enc[:,j,k]
                        reshaped_img_enc_pos[c] = [i, j, k]
                        ######## save the location as tuple
                        ######## dictionary or 2nd list
                        c += 1
            #### define dictionary of output
    
    ####### TRAIN BIRCH WITH 25% dataset
    ####### USE K_MEAN MODEL TO FIT IN DECISION TREE    

    ############## Training clustering model
    if training:
        print("Finish encoding data. Start Training...", flush=True)
        #### shape [i*24*24]
        # if clusterMethod == 0:
        #     start_time = time.time()
        #     cluster_model = affinity_progapation(reshaped_img_enc)
        #     print("--- Affinity Propagation: %s seconds ---" % (time.time() - start_time))
        #     model_name = "affinity_progapation.pkl"
        # if clusterMethod == 1:
        #     ### Decide number of K to clustering
        #     ### Distance: 0.6
        #     ### Loop for patches, if distance >.6, new cluster
            
        #     start_time = time.time()
        #     cluster_model = k_mean(reshaped_img_enc)
        #     print("--- K-mean : %s seconds ---" % (time.time() - start_time))
        #     model_name = "k_mean.pkl"
        # if clusterMethod == 2:
        #     start_time = time.time()
        #     cluster_model = mean_shift(reshaped_img_enc)
        #     print("--- Mean Shift: %s seconds ---" % (time.time() - start_time))
        #     model_name = "mean_shift.pkl"
        if clusterMethod == 3:
            #### compute the mean of the cluster
            #### use 64-dimension as well
            #### use 32-dimension as well
            #### use 16-dimension as well
            #### -->  
            eps = 0.40
            start_time = time.time()
            cluster_model = dbscan(reshaped_img_enc, eps=eps)
            print("--- DBScan: %s seconds ---" % (time.time() - start_time))
            model_name_json = "dbscan_core_" + str(skip_const) + "_" + str(eps) +".json"
            model_name = "dbscan_core_" + str(skip_const) + "_" + str(eps) +".pkl"
            core_sample_indices = cluster_model.core_sample_indices_
            components = cluster_model.components_
            
            
            result = dbscan_save(reshaped_img_enc, cluster_model.labels_, reshaped_img_enc_pos, imgs, core_sample_indices, components)
            path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/")
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path,model_name_json), 'w') as f:
                json.dump(result, f)
            save_model(cluster_model, os.path.join(path ,model_name))
        # if clusterMethod == 4:
        #     eps = 0.7
        #     start_time = time.time()
        #     cluster_model = optics(reshaped_img_enc, eps=eps)
        #     print("--- Optics: %s seconds ---" % (time.time() - start_time))
        #     model_name = "optics.pkl"
        # if clusterMethod == 5:
        #     # reshaped_img_enc = faiss_array(reshaped_img_enc, 128)
        #     start_time = time.time()
        #     cluster_model = birch(reshaped_img_enc, n_jobs=32)
        #     print("--- BIRCH(1-core): %s seconds ---" % (time.time() - start_time))
        #     model_name = "birch.pkl"
        
        
        # path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/")
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # save_model(cluster_model, os.path.join(path,model_name))
        # print("Finish training data.", flush=True)

    ############### Testing clustering model
    if not training:
        
        model_path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/")
        ##### LOAD DBSCAN MODEL USING JSON FILE
        # start_time = time.time()
        # embedded_vector, train_label, position, image_paths, thres = load_dbscan_json(model_path)
        # print("--- DBScan Load Model: %s seconds ---" % (time.time() - start_time))
        
        ###### LOAD DBSCAN MODEL USING PICKLE FILE
        start_time = time.time()
        embedded_vector, train_label, thres = load_dbscan_pickle(model_path)
        print("--- DBScan Load Model: %s seconds ---" % (time.time() - start_time))
        
        ### Predict DBScan label:
        start_time = time.time()
        test_label = batching_test_data(embedded_vector, train_label, reshaped_img_enc, thres)
        print("--- DBScan Prediction: %s seconds ---" % (time.time() - start_time))
        
        ### Load Decision Tree Model
        start_time = time.time()
        decision_model_name = "decision_tree_0.6.pkl"
        clf = load_model(os.path.join(model_path, decision_model_name))
        
        ### Predict using decision tree
        real_y = [ img[0].split("/")[-2] for img in imgs ]
        feature = clf.n_features_in_
        x = decision_tree_test_table(feature, test_label, reshaped_img_enc_pos, real_y)
        pred = clf.predict(x)
        score = clf.score(x, real_y)
        print("--- Decision Tree Prediction: %s seconds ---" % (time.time() - start_time))
        print("Prediction:", flush=True)
        print(pred, flush=True)
        print("Real Class:", flush=True)
        print(real_y, flush=True)
        print("Score: ", flush=True)
        print(score, flush=True)

        
        
        
        # unique, count = np.unique(cluster_model.labels_, return_counts=True)

        # groups = [[] for _ in range(len(unique))]
        # ##### label: [dataset_size*24*24]

        # for index, value in enumerate(test_label):
        #     i, j, k = reshaped_img_enc_pos[index]
        #     groups[value].append([i, j, k])
        
        # for i_group, group in enumerate(groups):
        #     group_dir = os.path.join(dir, str(i_group))
        #     if not os.path.exists(group_dir):
        #         os.makedirs(group_dir)
        #     else:
        #         shutil.rmtree(group_dir)
        #         os.makedirs(group_dir)

        #     for item in group:
        #         i = item[0]
        #         w = item[1]
        #         h = item[2]
        #         x = Image.open(imgs[i][0]).resize((224,224))
        #         x_tensor = transforms.ToTensor()(x).unsqueeze_(0) #shape (h, w)
        #         img_patch = x_tensor[0, :, w*8:min(224,w*8+patchsize),h*8:min(224,h*8+patchsize)]
        #         img_patch = transforms.ToPILImage()(img_patch)
        #         img_patch.save(os.path.join(group_dir, '%s_%s_%s.png'%(str(imgs[i][0].split('/')[-1].split('.png')[0]),str(w),str(h))))
        
    return
    
    

def save_model(model, path):
    sys.setrecursionlimit(50000)
    pickle.dump(model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

####### Fit training
####### Fit predict
# def affinity_progapation(input, **kwargs):
#     model = AffinityPropagation(random_state=5).fit(input)
#     return model

# def k_mean(input, **kwargs):
#     model = KMeans(n_clusters=9, random_state=0).fit(input)
#     return model

# def mean_shift(input, **kwargs):
#     model = MeanShift().fit(input)
#     return model

# def optics(input, eps, **kwargs):
#     output = OPTICS(min_samples=10, max_eps=eps, metric="euclidean", cluster_method="xi").fit(input)
#     return output

# def birch(input, **kwargs):
#     model = Birch(n_clusters=None).fit(input)
#     return model

# def decision_tree(model, cluster_label, cluster_label_pos, imgs):
#     ### Birch Model
#     unique, count = np.unique(model.labels_, return_counts=True)
#     d = len(unique)
#     n = len(imgs)
#     x = np.zeros((n, d))
#     y = imgs
#     for c in range(len(cluster_label)):
#         i, _, _ = cluster_label_pos[c]
#         j = cluster_label[c]
#         x[i, j] = 1
#     clf = DecisionTreeClassifier(random_state=0).fit(x, y)
#     return clf


############################### DBSCAN prediction:

### With 10 images, the higest matrix (float) will consume: 24(Bytes) * 360 * 1078920 * 128 /1024/1024/1024/1024 = 1.085(TB) in RAM.
### With input of 10 test images
### Input explanation:
# data: embedded vector of train data from BagNet
# label: label of embedded vector of training data from DBScan
# input_vector: embedded vector of test data from BagNet
# thres: 0.7

def batching_test_data(data, label, input_vector, thres):
    #### n: batching size (image)
    n = 2 ## 2 images
    k = int(input_vector.shape[0]/n)
    max_len = input_vector.shape[0]
    test_label = []
    for i in range(k):
        start = i*n
        if i == k-1:
            stop = max_len
        else:
            stop = (i+1)*n
        input = input_vector[start:stop, :]
        out_label = dbscan_prediction(data, label, input, thres)
        test_label.extend(out_label)
    return test_label
        

def dbscan_prediction(data, label, input_vector, thres):
    ## shape: (700000, 128)
    compareMatrix = np.array(data)
    ## shape: (36, 128)
    inputMatrix = np.array(input_vector)
    ## shape: (36, 700000, 128)
    z = compareMatrix - inputMatrix[:, None]
    ## shape: (36, 700000)
    distance = np.linalg.norm(z, axis=2)
    
    n = input_vector.shape[0]
    # shape(36)
    min_array = np.min(distance, axis=1)
    pos_array = np.argmin(distance, axis=1)
    out_label = [-1] * n
    for i in range(n):
        if min_array[i] < thres:
            index = pos_array[i]
            out_label[i] = label[index]
        else:
            out_label[i] = -1
    return out_label


def dbscan(input, eps, **kwargs):
    model = DBSCAN(eps=eps).fit(input)
    return model

def decision_tree_test_table(feature, label, position, imgs):
    d = feature
    n = len(imgs)
    x = np.zeros((n, d))
    for c in range(len(label)):
        i, _, _ = position[c]
        j = label[c]
        x[i, j] = 1
    return x

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
    y = [ img.split("/")[-2] for img in imgs ]
    x = decision_tree_table(label, position, unique_path)
    clf = DecisionTreeClassifier(random_state=0).fit(x, y)
    return clf


def dbscan_save(data, label, position, imgs, core_sample_indices, components):
    n = len(label)
    path = [None] * n
    c = 0
    for item in position:
        path[c] = imgs[item[0]][0]
        c = c + 1
    result = {"label": label.tolist(), "data": data.tolist(), "position":position, "path": path, "core_sample_indices": core_sample_indices.tolist(), "components": components.tolist()}
    return result


def dbscan_load(path):
    result = json.load(open(path, "rb"))
    return result

def outputing_train_image(label, position, imgs, dir, patchsize):
    
    groups = [[] for _ in range(max(label)+1)]
    ##### label: [dataset_size*24*24]
    
    for index, value in enumerate(label):
        if value != -1:
            i, j, k = position[index]
            groups[value].append([i, j, k])
        
    
    for i_group, group in enumerate(groups):
        group_dir = os.path.join(dir, str(i_group))
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        else:
            shutil.rmtree(group_dir)
            os.makedirs(group_dir)

        for item in group:
            i = item[0]
            w = item[1]
            h = item[2]
            x = Image.open(imgs[i][0]).resize((224,224))
            x_tensor = transforms.ToTensor()(x).unsqueeze_(0) #shape (h, w)
            img_patch = x_tensor[0, :, w*8:min(224,w*8+patchsize),h*8:min(224,h*8+patchsize)]
            img_patch = transforms.ToPILImage()(img_patch)
            img_patch.save(os.path.join(group_dir, '%s_%s_%s.png'%(str(imgs[i][0].split('/')[-1].split('.png')[0]),str(w),str(h))))
            
    return


def load_dbscan_json(model_path):
    #### Load DBScan Model:
    model_name = "dbscan_core_4_0.6.json"
    model = dbscan_load(os.path.join(model_path, model_name))
    ##### DECISION TREE CLASSIFIER
    embedded_vector = model["data"]
    train_label = model["label"]
    position = model["position"]
    image_paths = model["path"]
    thres = float(model_name.split("_")[-1][:3])
    return embedded_vector, train_label, position, image_paths, thres

def load_dbscan_pickle(model_path):
    ### Load DBScan Model:
    model_name = "dbscan_core_4_0.6.pkl"
    model = load_model(os.path.join(model_path, model_name))
    ### Decision Tree CLASSIFIER
    embedded_vector = model.components_
    train_label = model.labels_
    sample_indice = model.core_sample_indices_
    train_label = train_label[sample_indice]
    thres = float(model_name.split("_")[-1][:3])
    return embedded_vector, train_label, thres
    
    



    