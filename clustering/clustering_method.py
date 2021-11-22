from torch.utils.data import DataLoader
import numpy as np
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pickle
import joblib
import sys
import time
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift, DBSCAN, OPTICS, Birch

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
    D1 = len(imgs)
    D2 = 24
    D3 = 24
    reshaped_img_enc = np.empty([D1*D2*D3, input_channel])
    c = 0

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
            for k in range(img_shape[2]):
                reshaped_img_enc[c] = img_enc[:,j,k]
                ######## save the location as tuple
                ######## dictionary or 2nd list
                c += 1
        #### [dataset_size*24*24, 128]
    

    ############## Training clustering model
    if training:
        print("Finish encoding data. Start Training...", flush=True)
        #### shape [i*24*24]
        if clusterMethod == 0:
            cluster_model = affinity_progapation(reshaped_img_enc)
            model_name = "affinity_progapation.pkl"
        if clusterMethod == 1:
            start_time = time.time()
            cluster_model = k_mean(reshaped_img_enc)
            print("--- K-mean: %s seconds ---" % (time.time() - start_time))
            model_name = "k_mean.pkl"
        if clusterMethod == 2:
            cluster_model = mean_shift(reshaped_img_enc)
            model_name = "mean_shift.pkl"
        if clusterMethod == 3:
            eps = 0.6
            cluster_model = dbscan(reshaped_img_enc, eps=eps)
            model_name = "dbscan.pkl"
        if clusterMethod == 4:
            eps = 0.6
            labels = optics(reshaped_img_enc, eps=eps)
            # model_name = "optics.pkl"
        if clusterMethod == 5:
            start_time = time.time()
            cluster_model = birch(reshaped_img_enc)
            print("--- BIRCH: %s seconds ---" % (time.time() - start_time))
            model_name = "birch.pkl"
        
        path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/")
        if not os.path.exists(path):
            os.makedirs(path)
        save_model(cluster_model, os.path.join(path,model_name))
        print("Finish training data.", flush=True)
        return

    ###### Suitable Threshold: between 0.5->0.8
    ###### Change clustering method: 
    ###### Test with decision tree:



    ####### NEXT WEEK: Search for efficient clustering method
    ####### CONFIRM BAGNET WORKS OR NOT????????

    ############### Testing clustering model
    if not training:
        cluster_model = load_model(model_path)
        labels = cluster_model.predict(reshaped_img_enc)
        unique, count = np.unique(labels, return_counts=True)

        groups = [[] for _ in range(cluster_model.cluster_centers_.shape[0])]
        ##### label: [dataset_size*24*24]

        for index, value in enumerate(labels):
            i = int(int(index) / int(D2*D3))
            j = int((int(index) % int(D2*D3)) / int(D3))
            k = int((int(index) % int(D2*D3)) % int(D3))
            groups[value].append([i, j, k])
        
        for i_group, group in enumerate(groups[0:10]):
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

def save_model(model, path):
    sys.setrecursionlimit(10000)
    pickle.dump(model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

####### Fit training
####### Fit predict
def affinity_progapation(input, **kwargs):
    model = AffinityPropagation(random_state=5).fit(input)
    return model

def k_mean(input, **kwargs):
    model = KMeans(random_state=5).fit(input)
    return model

def mean_shift(input, **kwargs):
    model = MeanShift().fit(input)
    return model

def dbscan(input, eps, **kwargs):
    model = DBSCAN(eps=eps).fit(input)
    return model

def optics(input, eps, **kwargs):
    output = OPTICS(min_samples=10, max_eps=eps, metric="euclidean").fit_predict(input)
    return output

def birch(input, **kwargs):
    model = Birch(n_clusters=None).fit(input)
    return model