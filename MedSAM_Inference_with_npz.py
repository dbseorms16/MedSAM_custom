# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback
from PIL import Image
from skimage import io, transform
import cv2

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def finetune_model_predict(img_np, box_np, sam_trans, sam_model_tune, device='cuda:0'):
    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg

def resize_and_extract_bbox(gt2D):
    # gt2D = np.where(gt2D > 1, 1, 0)
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = np.array([x_min, y_min, x_max, y_max])
    
    return bbox
#%% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedSAM')
parser.add_argument('-i', '--data_path', type=str, default='data/Custom_data/Xerostomia/MRI', help='path to the data folder')
parser.add_argument('-iM', '--data_mask_path', type=str, default='data/Pre_embedding/test/npy_gts', help='path to the data folder')
parser.add_argument('-o', '--seg_path_root', type=str, default='data/Test_MedSAMBaseSeg', help='path to the segmentation folder')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--device', type=str, default='cuda:1', help='device')

parser.add_argument('--seg_png_path', type=str, default='data/sanity_test/Finetune_medsam_moving_boxes', help='path to the segmentation folder')
parser.add_argument('-chk', '--checkpoint', type=str, default='work_dir/SAM-ViT-B/sam_model_latest.pth', help='path to the trained model')
# parser.add_argument('-chk', '--checkpoint', type=str, default='work_dir/MedSAM/medsam_vit_b.pth', help='path to the trained model')
args = parser.parse_args()

#% load MedSAM model
device = args.device
sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

img_folders = sorted(os.listdir(args.data_path))
mask_folders = sorted(os.listdir(args.data_mask_path))

os.makedirs(args.seg_png_path, exist_ok=True)
save_path = join(args.seg_path_root, 'npy_embs')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

average_dice= []
num = 0
for img_folder, mask_folder in tqdm(zip(img_folders, mask_folders)):
    
    imgs = sorted(os.listdir(join(args.data_path, img_folder)))
    
    for img in imgs:
        img_np = np.array(Image.open(join(args.data_path, img_folder, img)).convert("L"))
        file1 = join(args.data_mask_path, 'Xerostomia_'+ img_folder +'_'+  img.split('.')[0] + '_1.npy')
        file2 = join(args.data_mask_path, 'Xerostomia_'+ img_folder +'_'+  img.split('.')[0] + '_2.npy')
        
        if not os.path.isfile(file1) or not os.path.isfile(file2) : continue
        gt2D_1 = np.load(file1)
        gt2D_2 = np.load(file2)
        
        ori_img = np.repeat(img_np[:, :, None], 3, axis=-1)
        ori_img_show = cv2.resize(ori_img.astype('float32'), (256,256)).astype('int64')
        
        sam_dice_scores = []
        bbox_1 = resize_and_extract_bbox(gt2D_1)
        bbox_2 = resize_and_extract_bbox(gt2D_2)
        # get bounding box from mask

        for i, (gt2D, bbox) in enumerate(zip([gt2D_1, gt2D_2], [bbox_1, bbox_2])):
            seg_mask = finetune_model_predict(ori_img, bbox, sam_trans, sam_model_tune, device=device)

            # these 2D dice scores are for debugging purpose. 
            # 3D dice scores should be computed for 3D images
            seg_mask = cv2.resize(seg_mask.astype('float32'), (512,512))
            dice = compute_dice(gt2D > 0, seg_mask > 0)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(ori_img)
            show_box(bbox, axes[0])
            show_mask(seg_mask, axes[0])
            axes[0].set_title('MedSAM: DSC={:.3f}'.format(dice))
            axes[0].axis('off')

            axes[1].imshow(ori_img)
            show_box(bbox, axes[1])
            show_mask(gt2D, axes[1])
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            # save figure
            fig.savefig(join(args.seg_png_path, str(img_folder)+ '_' + img.split('.tif')[0]+'_{}.png'.format(i+1)))
            num += 1
            average_dice.append(dice)
            # close figure
            plt.close(fig)
os.mkdir(join(args.seg_png_path, "{:.5f}".format(sum(average_dice) / len(average_dice))))