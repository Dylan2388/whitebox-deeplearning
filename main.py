import os
import time
import math
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
# import torch_xla.core.xla_model as xm

from util.log import Log
from util.data import get_dataloaders
from util.args import get_args, save_args, get_optimizer

from bagnet.bagnet import bagnet33, l2_norm_img
from bagnet.loss import patch_triplet_loss
from bagnet.visualize import show_triplets
from clustering.clustering_method import clustering

############################

def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    # model.eval()

#########################


def bagnet_process(training=True, visualize=False, cluster=True, cluster_training=True, cluster_testing=True):

    all_args = get_args()

    # Create a logger
    log = Log(all_args.log_dir)
    print("Log dir: ", all_args.log_dir, flush=True)
    # Log the run arguments
    save_args(all_args, log.metadata_dir)

    # set device
    if not all_args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.isdir(os.path.join(all_args.log_dir, "files")):
        os.mkdir(os.path.join(all_args.log_dir, "files")) 

    # Obtain the data loaders
    trainloader, projectloader, test_loader, classes, num_channels = get_dataloaders(all_args)

    # trained_orig_trees = []
    # trained_pruned_trees = []
    # trained_pruned_projected_trees = []
    # orig_test_accuracies = []
    # pruned_test_accuracies = []
    # pruned_projected_test_accuracies = []
    # project_infos = []
    # infos_sample_max = []
    # infos_greedy = []
    # infos_fidelity = []

    bagnet = bagnet33(device, pretrained=True)
    # unsupervised_layer = get_network(128, 'bagnet33', False)
    bagnet.to(device)

    # YOUR CODE HERE
    # parameters
    lr = 0.01
    epoch_num = 10
    
    optimizer_bagnet = optim.Adam(bagnet.parameters(), lr=lr)
    # freeze param
    for name, param in bagnet.named_parameters():
        if "unsup_layer" not in name:
            param.requires_grad = False

    # current directory
    directory = os.path.abspath(os.getcwd())
    model_path = os.path.join(directory, "bagnet/model/bagnet33.pth")
    
    if training:
        for i in range(0, epoch_num):

            if os.path.isfile(model_path):
                load_model(bagnet, model_path, device)
                print ("Load Model BagNet Successfully")

            # print epoch info
            c = 0
            p_old = 0
            print("Epoch " + str(i) + ":")


            for images, labels in trainloader:
                # forward
                images = images.to(device)
                # labels = labels.to(device)
                output = bagnet(images)
                # output = l2_norm_img(output)
                loss, dist_pc_pn, dist_pc_pf = patch_triplet_loss(output, 8, 0.2, 0.2)
                
                # backward
                optimizer_bagnet.zero_grad()
                loss.backward()
                optimizer_bagnet.step()

                ## Print progress
                p_new = round(c/len(trainloader), 2)
                if p_old != p_new:
                    print("### Progress: " + str(p_new))
                    p_old = p_new
                    c += 1
            save_model(bagnet, model_path, confirm=False)
            
    if visualize:
        load_model(bagnet, model_path, device)
        use_cosine = False
        if use_cosine:
            folder_name = "visualize_cosine"
        else:
            folder_name = "visualize_euclidean"
        show_triplets(bagnet, test_loader, folder_name, device, use_cosine, all_args)

    if cluster:
        load_model(bagnet, model_path, device)
        folder_name = "visualize_clustering"
        cluster_method = 2
        if cluster_training:
            clustering(bagnet, trainloader, folder_name, device, all_args, cluster_method, True)
        if cluster_testing:
            model_name = "mean_shift.pkl"
            model_path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/" + model_name)
            clustering(bagnet, test_loader, folder_name, device, all_args, cluster_method, False, model_path)


if __name__ == '__main__':
    bagnet_process(training=False, visualize=True, cluster=False, cluster_training=False, cluster_testing=True)