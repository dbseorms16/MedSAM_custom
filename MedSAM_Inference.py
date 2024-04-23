# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from PIL import Image

# np.set_printoptions(threshold=np.inf)
def calculate_iou(segmentation1, segmentation2):
    # segmentation1과 segmentation2는 두 개의 세그멘테이션 마스크 (이진 배열)입니다.
    # 각 세그멘테이션의 픽셀 수 계산
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    
    # IOU 계산
    iou = np.sum(intersection) / np.sum(union)
    
    return iou
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)

folder ="002"
file ="010"

parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    # default="assets/img_demo.png",
    default="data/Normal/MRI/{}/{}.tif".format(folder, file),
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    # default="assets/",
    default="data/Normal/Mask/{}/{}.tif".format(folder, file),
    help="path to the segmentation folder",
)
x = 100
x_pad = 100
y = 200
y_pad = 120

parser.add_argument(
    "--box",
    type=list,
    default=[x, y, x+x_pad, y+y_pad],
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()
# img_np = io.imread(args.data_path)
img_np = np.array(Image.open(args.data_path).convert("L"))
h_mask_np = np.array(Image.open(args.seg_path).convert("L"))

# 임계값 설정 (예: 128)
threshold = 1
# 이진화 수행

if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
    
h_mask_np = np.where(h_mask_np > threshold, 1, 0)
mask_np = np.zeros_like(h_mask_np)
b = args.box
# mask_np[b[0]:b[2], b[1]:b[3]] = h_mask_np[b[0]:b[2], b[1]:b[3]]
mask_np[:, 0:256] = h_mask_np[:, 0:256]
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

box_np = np.array([args.box])
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
# io.imsave(
#     join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
#     medsam_seg,
#     check_contrast=False,
# )

# %% visualize results
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img_3c)
show_box(box_np[0], ax[0])
ax[0].set_title("Input Image and Bounding Box")
ax[1].imshow(img_3c)
ax[2].imshow(img_3c)
show_mask(medsam_seg, ax[1])
show_box(box_np[0], ax[1])
show_mask(mask_np, ax[2])
show_box(box_np[0], ax[2])
ax[1].set_title("Prediction, iou {:3f}".format(calculate_iou(medsam_seg, mask_np)))
ax[2].set_title("GT")
plt.show()
