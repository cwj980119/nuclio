import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# COCO 데이터셋 경로 설정
coco_annotation_path = "path/to/coco_annotations.json"
coco_image_dir = "path/to/images/"
model_checkpoint = "sam_vit_b_01ec64.pth"

# COCO 파일 로드
with open(coco_annotation_path, "r") as f:
    coco_data = json.load(f)

# 이미지 정보와 어노테이션 매칭
images = {img["id"]: img for img in coco_data["images"]}
annotations = defaultdict(list)
for ann in coco_data["annotations"]:
    annotations[ann["image_id"]].append(ann)

# Bounding Box와 Mask 추출 함수
def extract_coco_data(image_id, annotations, image_dir):
    image_info = images[image_id]
    image_path = os.path.join(image_dir, image_info["file_name"])
    
    bboxes = []
    masks = []
    
    for ann in annotations[image_id]:
        # Bounding Box
        bbox = ann["bbox"]  # [x, y, width, height]
        x, y, w, h = bbox
        bboxes.append([x, y, x + w, y + h])
        
        # Segmentation Mask
        segm = ann["segmentation"]
        if isinstance(segm, list):  # Polygon format
            mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            for poly in segm:
                poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        else:  # RLE format
            from pycocotools import mask as mask_util
            mask = mask_util.decode(segm)
        masks.append(mask)
    
    return image_path, bboxes, masks

# SAM 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = sam_model_registry["vit_b"](model_checkpoint)
sam_model.to(device)
sam_model.train()

# 데이터 전처리
transform = ResizeLongestSide(sam_model.image_encoder.img_size)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])
    input_image = sam_model.preprocess(transformed_image)
    return input_image, original_image_size, input_size

# Optimizer 설정
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.BCELoss()

# 학습 루프
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    
    for image_id in tqdm(list(images.keys())[:20]):  # 첫 20개 이미지로 학습
        image_path, bboxes, masks = extract_coco_data(image_id, annotations, coco_image_dir)
        
        # 이미지 전처리
        input_image, original_image_size, input_size = preprocess_image(image_path)
        input_image = input_image.to(device)
        
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
        
        object_losses = []
        for bbox, gt_mask in zip(bboxes, masks):
            # Bounding Box 처리
            box = transform.apply_boxes(np.array(bbox), original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
            
            # Prompt Encoder
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            # Mask Decoder
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Mask Postprocessing
            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            predicted_mask = torch.sigmoid(upscaled_masks)
            
            # Ground Truth Mask
            gt_mask_resized = torch.from_numpy(
                np.resize(gt_mask, (1, 1, gt_mask.shape[0], gt_mask.shape[1]))
            ).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            
            # Loss 계산
            loss = loss_fn(predicted_mask, gt_binary_mask)
            object_losses.append(loss)
        
        # 각 이미지의 객체별 평균 Loss로 업데이트
        if object_losses:
            mean_object_loss = torch.mean(torch.stack(object_losses))
            optimizer.zero_grad()
            mean_object_loss.backward()
            optimizer.step()
            epoch_losses.append(mean_object_loss.item())
    
    # Epoch 손실 기록
    losses.append(epoch_losses)
    print(f"EPOCH: {epoch}, Mean Loss: {np.mean(epoch_losses) if epoch_losses else 0}")
