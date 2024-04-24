#%% import packages
# precompute image embeddings and save them to disk for model training

import numpy as np
import os
join = os.path.join 
from skimage import io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from PIL import Image

#%% parse arguments
# Normal
# Sialadenitis
# Xerostomia
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='./data/Custom_data/Xerostomia', help='# and also Tr_Release_Part2 when part1 is done')
parser.add_argument('-o', '--save_path', type=str, default='./data/Pre_embedding', help='path to save the image embeddings')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='/root/work_dir/MedSAM/medsam_vit_b.pth', help='path to the pre-trained SAM model')
args = parser.parse_args()

# machine = sorted(os.listdir(args.img_path))

pre_img_path = join(args.img_path, 'MRI') 
pre_mask_path = join(args.img_path, 'Mask') 
save_img_emb_path = join(args.save_path, 'npy_embs')
save_img_mask_path = join(args.save_path, 'img_mask')
save_gt_path = join(args.save_path, 'npy_gts')
os.makedirs(save_img_emb_path, exist_ok=True)
os.makedirs(save_gt_path, exist_ok=True)
folders = sorted(os.listdir(pre_img_path))
#%% set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to('cuda:0')
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

# mask_np[:, 0:256] = h_mask_np[:, 0:256]

# compute image embeddings
for folder in tqdm(folders):
    npz_files = sorted(os.listdir(join(pre_img_path, folder)))
    for name in tqdm(npz_files):
        img_G = np.array(Image.open(join(pre_img_path, folder, name)).convert("L"))
        if not img_G.shape == (512,512):
            img_G = img_G[40:,:] 
        img = np.repeat(img_G[:, :, None], 3, axis=-1)
        
        h_mask_np = np.array(Image.open(join(pre_mask_path, folder, name)).convert("L"))
        h_mask_np = np.where(h_mask_np > 1, 1, 0)
        gt1 = np.zeros_like(h_mask_np)
        gt2 = np.zeros_like(h_mask_np)
        # gt[:, 0:256] = h_mask_np[:, 0:256]
        gt1[:, 0:256] = h_mask_np[:, 0:256]
        gt2[:, 256:] = h_mask_np[:, 256:]
        
        resize_img = sam_transform.apply_image(img)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:0')
        # model input: (1, 3, 1024, 1024)
        input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
        
        a, b = np.where(gt1 > 0)
        if not a.size == 0:
            np.save(join(save_img_emb_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0]+'_1.npy'), embedding.cpu().numpy()[0])
            np.save(join(save_gt_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0]+'_1.npy'), gt1)
            img_idx1 = img.copy()
            bd1 = segmentation.find_boundaries(gt1, mode='inner')
            img_idx1[bd1, :] = [255, 0, 0]
            io.imsave(join(save_img_mask_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0] + '_1.png'), img_idx1)

        c, d = np.where(gt2 > 0)
        if not c.size == 0:
            np.save(join(save_img_emb_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0]+'_2.npy'), embedding.cpu().numpy()[0])
            np.save(join(save_gt_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0]+'_2.npy'), gt2)
        
            # sanity check
            img_idx2 = img.copy()
            bd2 = segmentation.find_boundaries(gt2, mode='inner')
            img_idx2[bd2, :] = [255, 0, 0]
            io.imsave(join(save_img_mask_path, args.img_path.split('/')[-1]+ '_' + folder + '_' + name.split('.tif')[0] + '_2.png'), img_idx2)
        
        