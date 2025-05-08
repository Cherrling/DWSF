import os
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from networks.segmentation.model import U2NETP
from utils.seg import init, generate_mask, pad_split_seg_rectify, obtain_wm_blocks, rectify
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import math

# 禁用 warning
import warnings
warnings.filterwarnings("ignore")

# 设置参数
parser = argparse.ArgumentParser(description='DWSF watermark segmentation evaluation')
parser.add_argument('--watermark_dir', type=str, required=True, help='Directory containing watermarked images')
parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing ground truth masks')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualization results')
parser.add_argument('--model_path', type=str, required=True, help='Path to the segmentation model')
parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for binary segmentation')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'blocks'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 初始化模型（会在generate_mask首次调用时自动初始化）
# 但我们可以在这里手动设置模型路径
model_path = args.model_path

init(model_path=model_path)

# 获取文件列表
watermark_files = sorted([f for f in os.listdir(args.watermark_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

# 确保文件数量一致
assert len(watermark_files) == len(mask_files), "Number of watermarked images and masks must be the same"

# 评估指标
total_iou = 0
total_dice = 0
total_precision = 0
total_recall = 0
total_f1 = 0

# 同时提取和评估水印块
total_blocks = 0
correct_blocks = 0

# 遍历所有图像进行评估
for i in tqdm(range(len(watermark_files)), desc="Evaluating"):
    # 读取图像和掩码
    img_path = os.path.join(args.watermark_dir, watermark_files[i])
    mask_path = os.path.join(args.mask_dir, mask_files[i])
    
    # 读取图像
    image = Image.open(img_path).convert('RGB')
    original_size = (image.height, image.width)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 读取真实掩码
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (original_size[1], original_size[0]))
    gt_mask = gt_mask / 255.0 if gt_mask.max() > 1 else gt_mask  # 确保范围在[0,1]
    gt_mask_binary = (gt_mask > 0.5).astype(np.float32)
    
    # 使用DWSF方法生成分割掩码
    with torch.no_grad():

        watermark_blocks, full_mask, original_mask, angle = obtain_wm_blocks(
            img_tensor,
            targetH=512,
            targetW=512,
            return_mask=True
        )

        pred_mask = full_mask.squeeze().cpu().numpy()    
        pred_mask_binary = (pred_mask > args.threshold).astype(np.float32)
    
    # 保存水印块以供检查
    for j, block in enumerate(watermark_blocks):
        block_np = ((block.cpu().numpy() + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'blocks', f'{os.path.splitext(watermark_files[i])[0]}_block{j}.png'), 
                   cv2.cvtColor(block_np, cv2.COLOR_RGB2BGR))
        total_blocks += 1
        
        # 检查块是否与真实掩码有重叠（简单评估提取的准确性）
        # 将块放回原图位置会比较复杂，这里简单统计块数量
    
    # 计算评估指标
    # IoU (Intersection over Union)
    intersection = np.logical_and(pred_mask_binary, gt_mask_binary).sum()
    union = np.logical_or(pred_mask_binary, gt_mask_binary).sum()
    iou = intersection / (union + 1e-6)
    
    # Dice系数
    dice = (2 * intersection) / (pred_mask_binary.sum() + gt_mask_binary.sum() + 1e-6)
    
    # 精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_mask_binary.flatten(), 
        pred_mask_binary.flatten(), 
        average='binary', 
        zero_division=0
    )
    # print(f"IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    # 累计指标
    total_iou += iou
    total_dice += dice
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    
    # 保存掩码
    cv2.imwrite(os.path.join(args.output_dir, 'masks', f'{os.path.splitext(watermark_files[i])[0]}_pred.png'), 
               (pred_mask_binary * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, 'masks', f'{os.path.splitext(watermark_files[i])[0]}_gt.png'), 
               (gt_mask_binary * 255).astype(np.uint8))
    
    # 创建可视化对比图
    img_np = np.array(image)
    
    # 创建热力图
    pred_heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gt_heatmap = cv2.applyColorMap((gt_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 创建透明叠加
    alpha = 0.5
    overlay_pred = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1 - alpha, pred_heatmap, alpha, 0)
    overlay_gt = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1 - alpha, gt_heatmap, alpha, 0)
    
    # 水平拼接图像
    h, w = img_np.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    canvas[:, w:2*w] = overlay_pred
    canvas[:, 2*w:] = overlay_gt
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "Prediction", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "Ground Truth", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"IoU: {iou:.4f}", (w + 10, 70), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"Dice: {dice:.4f}", (w + 10, 110), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"Precision: {precision:.4f}", (w + 10, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"Recall: {recall:.4f}", (w + 10, 190), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"F1: {f1:.4f}", (w + 10, 230), font, 1, (255, 255, 255), 2)
    
    # 保存对比图
    cv2.imwrite(os.path.join(args.output_dir, 'vis', f'{os.path.splitext(watermark_files[i])[0]}_comparison.png'), canvas)

# 计算平均指标
num_images = len(watermark_files)
avg_iou = total_iou / num_images
avg_dice = total_dice / num_images
avg_precision = total_precision / num_images
avg_recall = total_recall / num_images
avg_f1 = total_f1 / num_images

# 打印结果
print("\n===== DWSF Segmentation Evaluation Results =====")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice: {avg_dice:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Total extracted blocks: {total_blocks}")

# 保存评估结果
with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
    f.write("===== DWSF Segmentation Evaluation Results =====\n")
    f.write(f"Average IoU: {avg_iou:.4f}\n")
    f.write(f"Average Dice: {avg_dice:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")
    f.write(f"Average Recall: {avg_recall:.4f}\n")
    f.write(f"Average F1 Score: {avg_f1:.4f}\n")
    f.write(f"Total extracted blocks: {total_blocks}\n")

print(f"\nEvaluation completed. Results saved to {args.output_dir}")