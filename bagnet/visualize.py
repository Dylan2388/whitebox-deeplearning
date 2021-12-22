from torch.utils.data import DataLoader
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

def show_triplets(model, dataloader: DataLoader, foldername: str, device, use_cosine: bool, args: argparse.Namespace):
    # TODO make for other networks where step size != 8
    modelname = args.net

    if 'bagnet' not in modelname: #modelname can be "bagnet33" leading to patchsize = 33
        raise ValueError
    else:
        patchsize = modelname[-2:]
        if not patchsize.isdigit(): #if receptivefield < 10
            patchsize = modelname[-1:]
    patchsize = int(patchsize)
    #TODO make for other image sizes than 224
    model.eval()
    dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(dataloader))
        a = dataloader.dataset
        xs, ys = xs.to(device), ys.to(device)
        
        # Perform a forward pass through the network
        img_enc = model.forward(xs)

        for p in range(200):
            near_imgs_dir = os.path.join(dir, str(p))
            if not os.path.exists(near_imgs_dir):
                os.makedirs(near_imgs_dir)
            else:
                shutil.rmtree(near_imgs_dir)
                os.makedirs(near_imgs_dir)
            selection = torch.randint(low=8,high=img_enc.shape[2]-8,size=(2,))
            if p >= img_enc.shape[0]:
                p = p - img_enc.shape[0]
            
            nearest_patches = find_similar(img_enc[p,:,selection[0],selection[1]].unsqueeze_(1).unsqueeze_(2), model, dataloader, device, args, use_cosine)
            
            for (pn, patch_idx, similarity) in nearest_patches:
                x = Image.open(pn).resize((224,224)) #TODO make non hard coded
                x_tensor = transforms.ToTensor()(x).unsqueeze_(0) #shape (h, w)
                img_patch = x_tensor[0,:,patch_idx[0]*8:min(224,patch_idx[0]*8+patchsize),patch_idx[1]*8:min(224,patch_idx[1]*8+patchsize)] 
                img_patch = transforms.ToPILImage()(img_patch)
                img_patch.save(os.path.join(near_imgs_dir, '%s_%s.png'%(str(f"{similarity:.3f}"), str(pn.split('/')[-1].split('.png')[0]))))


# given a certain patch, find patches in the dataset which are similar (in this case: cosine similarity > 0.9)
def find_similar(current_patch, model, test_loader: DataLoader, device, args: argparse.Namespace, use_cosine: bool):
    nearest_patches = []
    sim_threshold = 0.9
    dist_threshold = 0.6
    imgs = test_loader.dataset.imgs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_normalize = transforms.Compose([
                            transforms.Resize(size=(224,224)),
                            transforms.ToTensor(),
                            normalize
                        ])
    ################ Do the same for the clustering process
    for i in imgs:
        img = Image.open(i[0])
        # skip image with 1 channel
        if len(img.getbands()) != 3:
            img = img.convert(mode='RGB')
        img_normalized_tensor = transform_normalize(img).unsqueeze_(0).to(device)
        # print("img normalized tensor: ", img_normalized_tensor.shape, img_normalized_tensor[0,:,:,:].shape)
        # Perform a forward pass through the network
        with torch.no_grad():
            # l2_norm_img
            img_enc = model.forward(img_normalized_tensor)
        
        if use_cosine:
            img_enc = img_enc.squeeze(0)
            # this implements cosine similarity, can be replaced with euclidean distance
            sim = F.cosine_similarity(current_patch,img_enc, dim = 0)
            max_sim_h, max_sim_h_idxs = torch.max(sim, dim=0)
            max_sim, max_sim_w_idx = torch.max(max_sim_h, dim=0)
            nearest_patch_idx = (max_sim_h_idxs[max_sim_w_idx].item(), max_sim_w_idx.item())
            if max_sim > sim_threshold:
                nearest_patches.append((i[0], nearest_patch_idx,max_sim.item()))
        else:
            dist = get_euclidean_distances(current_patch.unsqueeze(0), img_enc).squeeze()
            min_dist_h, min_dist_h_idxs = torch.min(dist, dim=0)
            min_dist, min_dist_w_idx = torch.min(min_dist_h, dim=0)
            nearest_patch_idx = (min_dist_h_idxs[min_dist_w_idx].item(), min_dist_w_idx.item())
            if min_dist < dist_threshold:
                nearest_patches.append((i[0], nearest_patch_idx,min_dist.item()))
    print("# near: ", len(nearest_patches), flush=True)
    return nearest_patches[:50]




####### Normalize the euclidean distance



def get_euclidean_distances(patches_c, xs):
    # Adapted from ProtoPNet
    # Computing ||xs - ps ||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
    # where ps is some prototype image
    # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
    # with weights set to 1 so each patch just has its values summed)
    ones = torch.ones_like(patches_c,
                            device=xs.device)  # Shape: (num_prototypes, num_features, w_1, h_1)
    xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)  # Shape: (bs, num_prototypes, w_in, h_in)
    # Now compute ||ps||^2
    # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
    # squared L2 distance
    ps_squared_l2 = torch.sum(patches_c ** 2,
                                dim=(1, 2, 3))  
    # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
    ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)
    # Compute xs * ps (for all patches in the input image)
    xs_conv = F.conv2d(xs, weight=patches_c) 
    # Use the values to compute the squared L2 distance
    distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
    distance = torch.sqrt(torch.abs(distance)+1e-14) #L2 distance (not squared). Small epsilon added for numerical stability
    return distance  




