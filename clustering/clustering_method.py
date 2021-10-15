from torch.utils.data import DataLoader
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AffinityPropagation

# 1. get output from bagnet 128-D vector (patch - cluster by patch)
# 2. feed the vector to clustering method
# 3. Get the result
# 4. trace back to the input, grouping them together

def clustering(model, dataLoader: DataLoader, device, args: argparse.Namespace, clusterMethod):
    modelname = args.net

    if 'bagnet' not in modelname: #modelname can be "bagnet33" leading to patchsize = 33
        raise ValueError
    else:
        patchsize = modelname[-2:]
        if not patchsize.isdigit(): #if receptivefield < 10
            patchsize = modelname[-1:]

    patchsize = int(patchsize)

    model.eval()

    # dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), foldername)
    # if not os.path.exists(dir):
    #     os.makedirs(dir)

    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(test_loader))
        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        img_enc = model.forward(xs)
        # [64, 128, 24, 24]
        img_shape = img_enc.shape
        reshaped_img_enc = torch.empty((img_shape[0]*img_shape[2]*img_shape[3], img_shape[1]), device=device)
        c = 0
        for i in range(img_shape[0]):
            for j in range(img_shape[2]):
                for k in range(img_shape[3]):
                    reshaped_img_enc[c] = img_enc[i,:,j,k]
                    c += 1

        cluster_labels = affinity_progapation(reshaped_img_enc)
        
        


def affinity_progapation(input, **kwargs):
    clustering = AffinityPropagation(random_state=5).fit(input)
    return clustering.labels_