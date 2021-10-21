from torch.utils.data import DataLoader
import numpy as np
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift

# 1. get output from bagnet 128-D vector (patch - cluster by patch)
# 2. feed the vector to clustering method
# 3. Get the result
# 4. trace back to the input, grouping them together

####### CLUSTER METHOD
### 0. Affinity propagation

def clustering(model, dataLoader: DataLoader, foldername: str, device, args: argparse.Namespace, clusterMethod: int):
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
    reshaped_img_enc = np.empty([D1*D2*D3, 128])
    c = 0

    for i, image in enumerate(imgs):
        img = Image.open(image[0])
        if len(img.getbands()) != 3:
            img = img.convert(mode='RGB')
        img_normalized_tensor = transform_normalize(img).unsqueeze_(0).to(device)


        with torch.no_grad():
            img_enc = model.forward(img_normalized_tensor)
            # [64, 128, 24, 24]
            # [64, 1, 24, 24]
        img_enc = img_enc.squeeze(0)
        img_shape = img_enc.shape

        # img_id = i*24*24 + j*24 + k
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                reshaped_img_enc[c] = img_enc[:,j,k]
                c += 1

    #### shape [i*24*24]
    if clusterMethod == 0:
        cluster_labels = affinity_progapation(reshaped_img_enc)
    if clusterMethod == 1:
        cluster_labels = k_mean(reshaped_img_enc)
    if clusterMethod == 2:
        cluster_labels = mean_shift(reshaped_img_enc)

    
    unique, count = np.unique(cluster_labels, return_counts=True)

    ###### create array group
    groups = [[]] * len(unique)

    for index, value in enumerate(cluster_labels):
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


####### Fit training
####### Fit predict
def affinity_progapation(input, **kwargs):
    clustering = AffinityPropagation(random_state=5).fit(input)
    return clustering.labels_

def k_mean(input, **kwargs):
    clustering = KMeans(random_state=5).fit(input)
    return clustering.labels_

def mean_shift(input, **kwargs):
    clustering = MeanShift().fit(input)
    return clustering.labels_