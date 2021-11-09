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

class EarlyStopping:
    def __init__(self, tolerance, patience):
        """
        Args:
          patience (int):    Maximum number of epochs with unsuccessful updates.
          tolerance (float): We assume that the update is unsuccessful if the validation error is larger
                              than the best validation error so far plus this tolerance.
        """
        self.tolerance = tolerance
        self.patience = patience
    
    def stop_criterion(self, val_errors, device):
        """
        Args:
          val_errors (iterable): Validation errors after every update during training.
        
        Returns: True if training should be stopped: when the validation error is larger than the best
                  validation error obtained so far (with given tolearance) for patience epochs (number of consecutive epochs for which the criterion is satisfied).
                 
                 Otherwise, False.
        """
        if len(val_errors) <= self.patience:
            return False

        min_val_error = min(val_errors)
        val_errors = np.array(val_errors[-self.patience:])
        return all(val_errors > min_val_error + self.tolerance)

#########################


def bagnet_process(training=True, visualize=False, visualize_trainloader=True, cluster=True, cluster_training=True, cluster_testing=True):

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

    # Number of output channel
    out_channel = 64
    bagnet = bagnet33(device, pretrained=True, out_channel=out_channel)
    bagnet.to(device)

    # YOUR CODE HERE
    # parameters
    lr = 0.001
    epoch_num = 50
    
    optimizer_bagnet = optim.Adam(bagnet.parameters(), lr=lr, weight_decay=0.0001)
    # early stoping initial
    train_errors = []  # Keep track of the training error
    val_errors = []  # Keep track of the validation error
    early_stop = EarlyStopping(tolerance=0.001, patience=5)
    # freeze param
    for name, param in bagnet.named_parameters():
        if "unsup_layer" not in name:
            param.requires_grad = False

    # current directory
    directory = os.path.abspath(os.getcwd())
    model = "bagnet/model/bagnet33_encoder" + str(out_channel) + ".pth"
    model_path = os.path.join(directory, model)
    
    if training:
        for epoch in range(epoch_num):

            if os.path.isfile(model_path):
                load_model(bagnet, model_path, device)
                print ("Load Model BagNet Successfully")


            ##### PRINT EPOCH:
            max_dist_pc_pn = -np.inf
            max_dist_pc_pf = -np.inf
            min_dist_pc_pn = np.inf
            min_dist_pc_pf = np.inf

            sum_dist_pc_pn = 0
            sum_dist_pc_pf = 0
            sum_loss = 0

            print("##################################", flush=True)
            print("EPOCH {:d}:".format(epoch), flush=True)
            
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

                ####### PRINT DISTANCE:
                # Sort ascending
                sorted_dist_pc_pn, _ = torch.sort(dist_pc_pn)
                sorted_dist_pc_pf, _ = torch.sort(dist_pc_pf)
                # Replace max
                if max_dist_pc_pn < sorted_dist_pc_pn[-1]:
                    max_dist_pc_pn = sorted_dist_pc_pn[-1].item()
                if max_dist_pc_pf < sorted_dist_pc_pf[-1]:
                    max_dist_pc_pf = sorted_dist_pc_pf[-1].item()
                # Replace min
                if min_dist_pc_pn > sorted_dist_pc_pn[0] and sorted_dist_pc_pn[0] > 0.0001:
                    min_dist_pc_pn = sorted_dist_pc_pn[0].item()
                if min_dist_pc_pf > sorted_dist_pc_pf[0] and sorted_dist_pc_pf[0] > 0.0001:
                    min_dist_pc_pf = sorted_dist_pc_pf[0].item()
                # Add sum
                sum_dist_pc_pn = sum_dist_pc_pn + torch.mean(dist_pc_pn).item()
                sum_dist_pc_pf = sum_dist_pc_pf + torch.mean(dist_pc_pf).item()
                sum_loss = sum_loss + loss.item()

            # Compute mean
            n = len(trainloader)
            mean_dist_pc_pn = sum_dist_pc_pn / n
            mean_dist_pc_pf = sum_dist_pc_pf / n
            mean_loss = sum_loss / n

            ####### PRINT INFO:
            print("Near Patch Distance:", flush=True)
            print("Max: {:.3f}".format(max_dist_pc_pn), flush=True)
            print("Min: {:.3f}".format(min_dist_pc_pn), flush=True)
            print("Mean: {:.3f}".format(mean_dist_pc_pn), flush=True)
            print("---")
            print("Far Patch Distance:", flush=True)
            print("Max: {:.3f}".format(max_dist_pc_pf), flush=True)
            print("Min: {:.3f}".format(min_dist_pc_pf), flush=True)
            print("Mean: {:.3f}".format(mean_dist_pc_pf), flush=True)
            print("---")
            print("Loss: {:.3f}".format(mean_loss), flush=True)

            ### SAVE MODEL
            save_model(bagnet, model_path, confirm=False)

            ### EARLY STOPPING
            train_errors.append(mean_loss)
            if early_stop.stop_criterion(train_errors, device):
                print(val_errors[epoch])
                print('STOP AFTER {:d} EPOCHS'.format(epoch))
                break
            

    if visualize:
        if visualize_trainloader:
            data = trainloader
            folder_name = "visualize_trainloader"
        else: 
            data = test_loader
            folder_name = "visualize_testloader"
        load_model(bagnet, model_path, device)
        use_cosine = False
        # if use_cosine:
        #     folder_name = "visualize_cosine"
        # else:
        #     folder_name = "visualize_euclidean"
        show_triplets(bagnet, data, folder_name, device, use_cosine, all_args)
        print("Finish Visualization")
        print("#################################")

    if cluster:
        load_model(bagnet, model_path, device)
        folder_name = "visualize_clustering"
        cluster_method = 4
        if cluster_training:
            clustering(bagnet, out_channel, trainloader, folder_name, device, all_args, cluster_method, True)
        if cluster_testing:
            model_name = "mean_shift.pkl"
            model_path = os.path.join(os.path.abspath(os.getcwd()), "clustering/model/" + model_name)
            clustering(bagnet, out_channel, test_loader, folder_name, device, all_args, cluster_method, False, model_path)


if __name__ == '__main__':
    bagnet_process(training=False, visualize=False, visualize_trainloader=True, cluster=True, cluster_training=True, cluster_testing=False)