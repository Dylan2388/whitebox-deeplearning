import torch
import torch.nn.functional as F

##################################################

import random

#inputs: a batch of image encodings (i.e. the output of bagnet, possibly after the extra 1x1 conv layer)
#distance should be at least 8 (meaning 8 to the right and 8 to the left) or larger. Since BagNet patches have overlap (with stride of 8), you are sure that a far patch is at least 8*8=64 pixels away (corresponding to ~2 patches of 33 pixels) and therefore is far enough from the current patch. Distance can also be larger, as long as 2*distance < size of the latent embedding. 
# I introduced two margins: margin_difficult and margin_difference. In the paper of Danon et al. they do not make this difference. You can give these margins the same value.
def patch_triplet_loss(img_enc, distance, margin_difficult, margin_difference):
    selection = torch.randint(low=0,high=img_enc.shape[2],size=(2,))
    pc = img_enc[:,:,selection[0],selection[1]].unsqueeze(2).unsqueeze(3) #shape (bs, D, 1, 1)
   
    near_indices_h = range(max(0,selection[0]-1),min(selection[0]+1+1,img_enc.shape[2])) #assumes 3x3 grid
    selected_near_h = random.choice(near_indices_h)
    near_indices_w = list(range(max(0,selection[1]-1),min(img_enc.shape[3],selection[1]+1+1))) #assumes 3x3 grid
    if selected_near_h == selection[0]: #near patch cannot have same index as current patch pc
        near_indices_w.remove(selection[1])
    selected_near_w = random.choice(near_indices_w)

    pn = img_enc[:,:,selected_near_h,selected_near_w].unsqueeze(2).unsqueeze(3)

    dist_pc_img = torch.sqrt(torch.abs(torch.sum(torch.pow(img_enc - pc, 2), dim=1)) +1e-14)
    dist_pc_pn = dist_pc_img[:,selected_near_h,selected_near_w]

    mask = torch.ones_like(dist_pc_img[0,:,:])
    mask[max(0,selection[0]-distance):min(selection[0]+distance+1,img_enc.shape[2]),max(0,selection[1]-distance):min(img_enc.shape[3],selection[1]+distance+1)]=0.
  
    masked_near_dist = dist_pc_img * mask.unsqueeze(0).repeat(dist_pc_img.shape[0],1,1)

    mask_margin = dist_pc_img.le(margin_difficult).int().float()

    masked_masked_distances = masked_near_dist * mask_margin

    pf = torch.ones_like(pn)
    dist_pc_pf = torch.zeros_like(dist_pc_pn)
    for sample in range(dist_pc_img.shape[0]):
        nonzeros = torch.nonzero(masked_masked_distances[sample,:,:])
        if nonzeros.nelement()>0:
            selected_distant_idx = random.choice(nonzeros)
        else: #no distant patch within margin, just select random distant patch
            nonzeros_le_margin = torch.nonzero(masked_near_dist[sample,:,:])
            selected_distant_idx = random.choice(nonzeros_le_margin)
        dist_pc_pf[sample] = dist_pc_img[sample,selected_distant_idx[0],selected_distant_idx[1]]
 
        pf[sample,:,:]=img_enc[sample,:,selected_distant_idx[0],selected_distant_idx[1]].unsqueeze(1).unsqueeze(2)
        
    loss = torch.mean(F.relu(dist_pc_pn - dist_pc_pf + margin_difference))

    return loss, dist_pc_pn, dist_pc_pf


# normalizes image encoding to unit length
# adapted from https://github.com/tbmoon/facenet/blob/master/models.py#L70
def l2_norm_img(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.unsqueeze(1).expand_as(input))
    output = _output.view(input_size)
    return output

