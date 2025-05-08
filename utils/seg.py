# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import torch
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
import math
import os
from networks.segmentation.model import U2NETP

model = None
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
interpolatemode = 'bicubic'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def init(model_path=''):
    global model
    if model_path == '':
        # error
        raise ValueError("Please provide the model path")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        # error
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    model = U2NETP(mode='eval').to(device)
    checkpoint = torch.load(model_path)
    # checkpoint = torch.load('./results/seg_checkpoint/seg.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


def generate_mask(image):
    """
    return mask
    """
    if model is None:
        init()
    if isinstance(image, list):
        image = [F.interpolate(im, (512,512), mode=interpolatemode) for im in image]
        image = torch.vstack(image)
    else:
        image = F.interpolate(image, (512,512), mode=interpolatemode)
    with torch.no_grad():
        image = image.cuda()
        d0, d1, d2, d3, d4, d5, d6 = model(image)
    return d0


def rectify(image, mask, threshold=128):
    """
    rectify geometric transform
    """
    if isinstance(image,torch.Tensor):
        image = image.clone().detach_()
        image = image.cpu().numpy().transpose(0,2,3,1)[0]
        image = (image+1.)*127.5
    else:
        image = np.array(image)

    # binarize mask
    mask = mask[:, :, 0]
    mask = np.uint8(mask * 255)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    # find top20 minimum bounding rectangles
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    # estimate rotation angle
    angles = [0]
    for cnt in sorted_contours:
        rec = cv2.minAreaRect(cnt)
        width, height = rec[1]
        angle = rec[2]
        if width >= 55 and height >= 55 and width <= 4 * height and height <= 4 * width and height<=265 and width <= 265:
            if abs(angle) > 45:
                if angle < 0:
                    angle = -(angle + 90)
                else:
                    angle = 90 - angle
            else:
                angle = -angle
            angle = round(angle)
            angles.append(angle)
    angles = search_near(angles, [8,4,2])
    angle = np.mean(angles)

    # inverse rotation
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), -angle, 1)
    rotate_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    rotate_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # find top20 minimum bounding rectangles
    rotate_mask[rotate_mask < threshold] = 0
    rotate_mask[rotate_mask >= threshold] = 255
    _, thresh = cv2.threshold(rotate_mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    # obtain all blocks with defalut size (128x128)
    images_list = []
    height_list = []
    width_list = []
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h>= 55 and w >= 55 and w <= 4 * h and h <= 4 * w and h<=265 and w<=265:
            tmp = rotate_img[y:y + h, x:x + w]
            tmp = cv2.resize(tmp, (128, 128))
            images_list.append(tmp)
            height_list.append(h)
            width_list.append(w)


    if len(height_list) == 0:
        height_list.append(128)
    if len(width_list) == 0:
        width_list.append(128)
    height_list = search_near(height_list, [64,32,16,8])
    width_list = search_near(width_list, [64,32,16,8])

    if len(images_list) == 0:
        images_list.append(cv2.resize(rotate_img, (128,128)))

    images_list = [torch.tensor(image) for image in images_list]
    images_tensor = torch.stack(images_list)
    images_tensor = images_tensor.permute((0,3,1,2))
    images_tensor = (images_tensor/255-0.5)/0.5

    return images_tensor, [np.mean(height_list), np.mean(width_list)], angle


def search_near(num_list, qs_list=[]):
    """
    search near results within given quantization step
    """
    for qs in qs_list:
        num_list_bak = []
        for i in num_list:
            num_list_bak.append(i // qs * qs)
        maxlabel = max(set(num_list_bak), key=num_list_bak.count)
        result = []
        for i in range(len(num_list_bak)):
            if num_list_bak[i] == maxlabel:
                result.append(num_list[i])
        num_list = result
    return num_list


def pad_split_seg_rectify(image, targetH=512, targetW=512, return_mask=True):
    """
    segment and rectify
    
    Args:
        image: 输入图像张量 [B, C, H, W]
        targetH, targetW: 分块大小
        return_mask: 是否返回掩码信息
        
    Returns:
        image_tensor: 校正后的水印块
        [height, width]: 水印块尺寸
        angle: 校正角度
        full_mask: (可选)完整的分割掩码
        original_mask: (可选)原始未校正掩码
    """
    height, width = image.shape[2], image.shape[3]
    target_height = math.ceil(height/targetH)*targetH
    target_width = math.ceil(width/targetW)*targetW

    # pad, split and segment
    image_pad = F.pad(image, [0, target_width-width, 0, target_height-height], mode='constant', value=0)
    imagepatchs = image_pad.view(image_pad.shape[0], image_pad.shape[1], 
                                image_pad.shape[2]//targetH, targetH, 
                                image_pad.shape[3]//targetW, targetW)
    imagepatchs = imagepatchs.permute(2, 4, 0, 1, 3, 5)
    imagestensor = imagepatchs.reshape(-1, imagepatchs.shape[3], imagepatchs.shape[4], imagepatchs.shape[5])
    mask_list = []
    
    # 批量生成分割掩码
    for i in range(0, imagestensor.shape[0], 8):
        end_idx = min(i+8, imagestensor.shape[0])
        mask_tensor = generate_mask(imagestensor[i:end_idx])
        mask_list.append(mask_tensor)
    
    # 重组掩码
    mask_tensor = torch.vstack(mask_list)
    mask_tensor = mask_tensor.reshape(imagepatchs.shape[0], imagepatchs.shape[1], 
                                     imagepatchs.shape[2], 1, 
                                     imagepatchs.shape[4], imagepatchs.shape[5])
    mask_tensor = mask_tensor.permute(2, 3, 0, 4, 1, 5)
    mask_tensor = mask_tensor.reshape(image_pad.shape[0], 1, image_pad.shape[2], image_pad.shape[3])
    
    # 保存原始掩码(未经sigmoid激活)
    original_mask = mask_tensor.clone()
    
    # 转置并转为numpy进行校正
    mask_tensor_np = mask_tensor.permute(0, 2, 3, 1)
    mask = mask_tensor_np.cpu().numpy()

    # 校正并提取水印块
    image_tensor, [block_height, block_width], angle = rectify(image_pad, mask[0])
    
    # 如果需要返回掩码
    if return_mask:
        # 裁剪掩码到原始尺寸
        original_mask = original_mask[:, :, :height, :width]
        
        # 根据校正角度计算校正后的掩码
        # 将掩码转换为概率
        prob_mask = torch.sigmoid(mask_tensor)
        # 转为numpy进行旋转操作
        prob_mask_np = prob_mask.cpu().squeeze().numpy()
        # 旋转掩码(与图像旋转相同的角度)
        M = cv2.getRotationMatrix2D((prob_mask_np.shape[1] // 2, prob_mask_np.shape[0] // 2), -angle, 1)
        rotated_mask = cv2.warpAffine(prob_mask_np, M, (prob_mask_np.shape[1], prob_mask_np.shape[0]))
        # 裁剪到原始大小
        full_mask = rotated_mask[:height, :width]
        # 转回tensor
        full_mask = torch.from_numpy(full_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        
        return image_tensor.cuda(), [block_height, block_width], angle, full_mask, original_mask
    else:
        return image_tensor.cuda(), [block_height, block_width], angle


def obtain_wm_blocks(image, targetH=512, targetW=512, return_mask=False):
    """
    return rectified watermarked blocks and masks
    
    Args:
        image: 输入图像张量
        targetH, targetW: 分块大小
        return_mask: 是否返回掩码信息
        
    Returns:
        image_tensor: 校正后的水印块
        full_mask: (可选)完整的分割掩码
        original_mask: (可选)原始未校正掩码
        angle: (可选)校正角度
    """
    if return_mask:
        # 第一次处理
        image_tensor1, [height, width], angle, full_mask1, original_mask1 = pad_split_seg_rectify(
            image, targetH=targetH, targetW=targetW, return_mask=True)
        
        # 根据首次估计的水印块大小调整图像尺寸
        resized_image = F.interpolate(
            image, 
            (math.ceil(image.shape[2]/(height/128)), math.ceil(image.shape[3]/(width/128))),
            mode=interpolatemode)
        
        # 第二次处理
        image_tensor2, [height, width], angle, full_mask2, original_mask2 = pad_split_seg_rectify(
            resized_image, targetH=targetH, targetW=targetW, return_mask=True)
        
        # 调整第二个掩码到原始尺寸
        if full_mask2.shape[2:] != full_mask1.shape[2:]:
            full_mask2 = F.interpolate(full_mask2, size=full_mask1.shape[2:], mode='bilinear', align_corners=True)
            original_mask2 = F.interpolate(original_mask2, size=original_mask1.shape[2:], mode='bilinear', align_corners=True)
        
        # 合并结果(取最大值)
        image_tensor = torch.vstack([image_tensor1, image_tensor2])
        full_mask = torch.max(full_mask1, full_mask2)
        original_mask = torch.max(original_mask1, original_mask2)
        
        return image_tensor.cuda(), full_mask, original_mask, angle
    else:
        # 原始功能保持不变
        image_tensor1, [height, width], angle = pad_split_seg_rectify(image, targetH=targetH, targetW=targetW, return_mask=False)
        image = F.interpolate(image, (math.ceil(image.shape[2]/(height/128)), math.ceil(image.shape[3]/(width/128))), mode=interpolatemode)
        image_tensor2, [height, width], angle = pad_split_seg_rectify(image, targetH=targetH, targetW=targetW, return_mask=False)
        
        image_tensor = torch.vstack([image_tensor1, image_tensor2])
        return image_tensor.cuda()