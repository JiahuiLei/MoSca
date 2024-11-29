from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image
from torch import nn
import os, os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np


def get_segformer_model(device):
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return feature_extractor, model


def get_mask(img_np, feature_extractor, model, device, class_id=[2]):
    # image = Image.open(fn)
    # convert numpy img with uint8 to PIL img with uint8
    assert img_np.dtype == np.uint8
    image = Image.fromarray(img_np)
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    outputs = model(pixel_values)
    logits = outputs.logits
    logits = nn.functional.interpolate(
        outputs.logits.detach().cpu(),
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    # Second, apply argmax on the class dimension
    seg = logits.argmax(dim=1)[0]
    # class name
    mask_list = []
    for i in class_id:
        mask_list.append(seg == i)
    mask = torch.stack(mask_list, dim=0).float()
    mask = mask.any(dim=0).float()
    return mask


def segformer_sky_process_folder(feature_extractor, model, src, dst):
    device = next(model.parameters()).device
    os.makedirs(dst, exist_ok=True)
    fns = os.listdir(src)
    fns.sort()
    for fn in tqdm(fns):
        in_fn = os.path.join(src, fn)
        out_fn = os.path.join(dst, fn)
        mask = get_mask(in_fn, feature_extractor, model, device)
        mask = mask.cpu().numpy().astype(np.uint8)
        cv.imwrite(out_fn, mask * 255)
    return


def dummy_segformer_sky_process_folder(src, dst):
    os.makedirs(dst, exist_ok=True)
    fns = os.listdir(src)
    fns.sort()
    for fn in tqdm(fns):
        in_fn = os.path.join(src, fn)
        out_fn = os.path.join(dst, fn)
        img = cv.imread(in_fn)
        mask = np.zeros_like(img)
        cv.imwrite(out_fn, mask)
    return
