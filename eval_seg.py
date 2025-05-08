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
from utils.seg import init, generate_mask
from sklearn.metrics import precision_recall_fscore_support, jaccard_score

# 设置参数
parser = argparse.ArgumentParser(description='Evaluate watermark segmentation model')
parser.add_argument('--watermark_dir', type=str, required=True, help='Directory containing watermarked images')
parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing ground truth masks')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualization results')
parser.add_argument('--model_path', type=str, default='./seg_99.pth', help='Path to the segmentation model')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = U2NETP(mode='test').to(device)
checkpoint = torch.load(args.model_path)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 获取文件列表
watermark_files = sorted(os.listdir(args.watermark_dir))
mask_files = sorted(os.listdir(args.mask_dir))

# 确保文件数量一致
assert len(watermark_files) == len(mask_files), "Number of watermarked images and masks must be the same"

# 评估指标
total_iou = 0
total_dice = 0
total_precision = 0
total_recall = 0
total_f1 = 0

# 遍历所有图像进行评估
for i in tqdm(range(len(watermark_files)), desc="Evaluating"):
    # 读取图像和掩码
    img_path = os.path.join(args.watermark_dir, watermark_files[i])
    mask_path = os.path.join(args.mask_dir, mask_files[i])
    
    # 读取并预处理图像
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 读取真实掩码
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (512, 512))
    gt_mask = gt_mask / 255.0 if gt_mask.max() > 1 else gt_mask  # 确保范围在[0,1]
    gt_mask_binary = (gt_mask > 0.5).astype(np.float32)
    
    # 预测掩码
    with torch.no_grad():
        # 调整图像大小以适应模型
        img_resized = F.interpolate(img_tensor, (512, 512), mode='bicubic')
        # 预测
        d0, d1, d2, d3, d4, d5, d6 = model(img_resized)
        # 使用主输出
        pred_mask = torch.sigmoid(d0).squeeze().cpu().numpy()
        
    # 二值化预测掩码
    pred_mask_binary = (pred_mask > args.threshold).astype(np.float32)
    
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
    
    # 累计指标
    total_iou += iou
    total_dice += dice
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    
    # 创建可视化结果
    # 调整掩码大小为原始图像大小
    pred_mask_vis = cv2.resize(pred_mask, (image.width, image.height))
    gt_mask_vis = cv2.resize(gt_mask, (image.width, image.height))
    
    # 将预测掩码转换为彩色热力图
    pred_mask_color = cv2.applyColorMap((pred_mask_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gt_mask_color = cv2.applyColorMap((gt_mask_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 创建原始图像的副本
    img_np = np.array(image)
    
    # 创建透明叠加
    alpha = 0.5
    overlay_pred = cv2.addWeighted(img_np, 1 - alpha, pred_mask_color, alpha, 0)
    overlay_gt = cv2.addWeighted(img_np, 1 - alpha, gt_mask_color, alpha, 0)
    
    # 水平拼接原始图像、预测掩码和真实掩码
    h, w = img_np.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = img_np
    canvas[:, w:2*w] = overlay_pred
    canvas[:, 2*w:] = overlay_gt
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "Prediction", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "Ground Truth", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"IoU: {iou:.4f}", (w + 10, 70), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"Dice: {dice:.4f}", (w + 10, 110), font, 1, (255, 255, 255), 2)
    
    # 保存结果
    base_filename = os.path.splitext(watermark_files[i])[0]
    cv2.imwrite(os.path.join(args.output_dir, 'vis', f'{base_filename}_comparison.png'), canvas)
    
    # 也分别保存预测掩码和真实掩码
    pred_mask_bin_vis = (pred_mask_binary * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, f'{base_filename}_pred.png'), cv2.resize(pred_mask_bin_vis, (gt_mask.shape[1], gt_mask.shape[0])))
    cv2.imwrite(os.path.join(args.output_dir, f'{base_filename}_gt.png'), (gt_mask_binary * 255).astype(np.uint8))

# 计算平均指标
num_images = len(watermark_files)
avg_iou = total_iou / num_images
avg_dice = total_dice / num_images
avg_precision = total_precision / num_images
avg_recall = total_recall / num_images
avg_f1 = total_f1 / num_images

# 保存评估结果
results = {
    "IoU": avg_iou,
    "Dice": avg_dice,
    "Precision": avg_precision,
    "Recall": avg_recall,
    "F1": avg_f1
}

print("\n===== Segmentation Evaluation Results =====")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice: {avg_dice:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# 保存结果到文本文件
with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
    f.write("===== Segmentation Evaluation Results =====\n")
    f.write(f"Average IoU: {avg_iou:.4f}\n")
    f.write(f"Average Dice: {avg_dice:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")
    f.write(f"Average Recall: {avg_recall:.4f}\n")
    f.write(f"Average F1 Score: {avg_f1:.4f}\n")

print(f"\nEvaluation completed. Results saved to {args.output_dir}")