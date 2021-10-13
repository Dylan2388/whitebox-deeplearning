from torch.utils.data import DataLoader
import os, shutil
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

def show_triplets(model, test_loader: DataLoader, foldername: str, device, args: argparse.Namespace):
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
        xs, ys = next(iter(test_loader))
        xs, ys = xs.to(device), ys.to(device)
        
        # Perform a forward pass through the network
        img_enc = model.forward(xs)

        for p in range(5):
            near_imgs_dir = os.path.join(dir, str(p))
            if not os.path.exists(near_imgs_dir):
                os.makedirs(near_imgs_dir)
            else:
                shutil.rmtree(near_imgs_dir)
                os.makedirs(near_imgs_dir)
            selection = torch.randint(low=0,high=img_enc.shape[2],size=(2,))
            if p >= img_enc.shape[0]:
                p = p - img_enc.shape[0]
            
            nearest_patches = find_similar(img_enc[p,:,selection[0],selection[1]].unsqueeze_(1).unsqueeze_(2), model, test_loader, device, args)
            
            for (pn, patch_idx, similarity) in nearest_patches:
                x = Image.open(pn).resize((224,224)) #TODO make non hard coded
                x_tensor = transforms.ToTensor()(x).unsqueeze_(0) #shape (h, w)
                img_patch = x_tensor[0,:,patch_idx[0]*8:min(224,patch_idx[0]*8+patchsize),patch_idx[1]*8:min(224,patch_idx[1]*8+patchsize)] 
                img_patch = transforms.ToPILImage()(img_patch)
                img_patch.save(os.path.join(near_imgs_dir, '%s_%s.png'%(str(pn.split('Scene')[-1].split('.png')[0]),str(f"{similarity:.3f}"))))

# given a certain patch, find patches in the dataset which are similar (in this case: cosine similarity > 0.9)
def find_similar(current_patch, model, test_loader: DataLoader, device, args: argparse.Namespace):
    nearest_patches = []
    sim_threshold = 0.9
    imgs = test_loader.dataset.imgs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_normalize = transforms.Compose([
                            transforms.Resize(size=(224,224)),
                            transforms.ToTensor(),
                            normalize
                        ])
    for i in imgs:
        img = Image.open(i[0])
        img_normalized_tensor = transform_normalize(img).unsqueeze_(0).to(device)
        # print("img normalized tensor: ", img_normalized_tensor.shape, img_normalized_tensor[0,:,:,:].shape)
        # Perform a forward pass through the network
        with torch.no_grad():
            img_enc = model.forward(img_normalized_tensor)
        
        img_enc = img_enc.squeeze(0)
        # this implements cosine similarity, can be replaced with euclidean distance
        sim = F.cosine_similarity(current_patch,img_enc, dim = 0)
        max_sim_h, max_sim_h_idxs = torch.max(sim, dim=0)
        max_sim, max_sim_w_idx = torch.max(max_sim_h, dim=0)
        nearest_patch_idx = (max_sim_h_idxs[max_sim_w_idx].item(), max_sim_w_idx.item())
        if max_sim > sim_threshold:
            nearest_patches.append((i[0], nearest_patch_idx,max_sim.item()))
    print("# near: ", len(nearest_patches), flush=True)
    return nearest_patches[:50]