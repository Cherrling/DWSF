"""
独立测试EncoderDecoder模型的编码解码准确度
可测试不同攻击方式下的水印提取性能
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import kornia
import matplotlib.pyplot as plt

from networks.models.EncoderDecoder import EncoderDecoder
from networks.models.Noiser import Noise
from utils.util import generate_random_coor, setup_seed, save_images
from utils.img import psnr_clip
from utils.seg import obtain_wm_blocks, init
from utils.crc import crc

# 固定随机种子
setup_seed(42)

def encode(encoder, images, messages, splitSize=128, inputSize=128, h_coor=[], w_coor=[], psnr=35):
    """编码图像块，在随机坐标位置嵌入水印"""
    with torch.no_grad():
        if isinstance(messages, np.ndarray):
            messages = torch.Tensor(messages)
            messages = messages.to(device)
        
        # 获取图像块
        tmp_blocks = []
        for i in range(len(h_coor)):
            x1 = h_coor[i]-splitSize//2
            x2 = h_coor[i]+splitSize//2
            y1 = w_coor[i]-splitSize//2
            y2 = w_coor[i]+splitSize//2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                tmp_block = images[:, :, x1:x2, y1:y2]
                tmp_blocks.append(tmp_block)
        
        if len(tmp_blocks) == 0:
            return images
            
        tmp_blocks = torch.vstack(tmp_blocks)
        tmp_blocks_bak = tmp_blocks.clone()
        if splitSize != inputSize:
            tmp_blocks = F.interpolate(tmp_blocks, (inputSize, inputSize), mode='bicubic')
        
        # 编码图像块
        messages = messages.repeat((tmp_blocks.shape[0], 1))
        tmp_encode_blocks = encoder(tmp_blocks, messages)
        tmp_noise = tmp_encode_blocks - tmp_blocks
        tmp_noise = torch.clamp(tmp_noise, -0.2, 0.2)
        if splitSize != inputSize:
            tmp_noise = F.interpolate(tmp_noise, (splitSize, splitSize), mode='bicubic')

        # 将编码后的块合并到水印图像中
        watermarked_images = images.clone().detach_()
        for i in range(len(h_coor)):
            x1 = h_coor[i]-splitSize//2
            x2 = h_coor[i]+splitSize//2
            y1 = w_coor[i]-splitSize//2
            y2 = w_coor[i]+splitSize//2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                ori_block = tmp_blocks_bak[i:i+1, :, :, :]
                en_block = ori_block + tmp_noise[i:i+1, :, :, :]
                # en_block = psnr_clip(en_block, ori_block, psnr)
                watermarked_images[:, :, x1:x2, y1:y2] = en_block

        return watermarked_images

def decode(decoder, noised_images):
    """解码水印图像"""
    with torch.no_grad():
        # 获取水印块
        noised_blocks = obtain_wm_blocks(noised_images)
        if len(noised_blocks) == 0:
            # 如果没有检测到水印块，返回空结果
            return torch.zeros((1, message_length), device=device)
            
        # 分批解码，避免内存溢出
        decode_messages = []
        for i in range(0, len(noised_blocks), 32):
            decode_messages.append(decoder(noised_blocks[i:i+32]))
        decode_messages = torch.vstack(decode_messages)
    
        return decode_messages

def image_quality_evaluate(images, encoded_images):
    """评估图像质量（PSNR和SSIM）"""
    psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images, max_val=2.).item()
    ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, 
                                          max_val=1., window_size=5, reduction="mean").item()
    return psnr, ssim

def calc_bit_accuracy(original, decoded, threshold=0.5):
    """计算比特准确率"""
    decoded_binary = (decoded > threshold).float()
    correct = (decoded_binary == original).float().mean().item()
    return correct

def message_fusion(messages):
    """融合多个解码信息，提高准确率"""
    fusion_messages = torch.mean(messages, dim=0, keepdim=True)
    return fusion_messages

def visualize_results(results, attack_names, save_path=None):
    """可视化不同攻击下的结果"""
    metrics = ['PSNR', 'SSIM', 'Bit Accuracy', 'CRC Success Rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[attack][i] for attack in attack_names]
        ax.bar(attack_names, values)
        ax.set_title(metric)
        ax.set_ylim(0, 1.0 if i > 1 else (50 if i == 0 else 1.0))  # 适当设置y轴范围
        ax.set_xticklabels(attack_names, rotation=45, ha='right')
        
        # 在柱状图上方显示数值
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def crc_evaluate(ori_decode_messages):
    """评估CRC校验结果"""
    decode_messages = (ori_decode_messages.gt(0.5)).int()
    decode_messages = decode_messages.cpu().numpy()
    decode_messages[decode_messages != 0] = 1
    flag = 0
    for j in range(len(decode_messages)):
        flag = crc(decode_messages[j:j+1, :], 'decode')
        if flag == 1:
            break
    return flag

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试EncoderDecoder准确度')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image_dir', type=str, required=True, help='测试图像目录')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='输出目录')
    parser.add_argument('--seg_model', type=str, default='./seg_99.pth', help='分割模型路径')
    parser.add_argument('--use_crc', action='store_true', help='是否使用CRC校验')
    parser.add_argument('--message_length', type=int, default=30, help='水印信息长度')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--psnr_target', type=float, default=35, help='目标PSNR')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    return parser.parse_args()

def search_image_files(directory):
    """递归搜索目录中的所有图像文件"""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    """主函数"""
    args = parse_args()
    
    global device, message_length
    message_length = args.message_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化分割模型
    init(model_path=args.seg_model)
    print(f"分割模型已加载: {args.seg_model}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    watermarked_dir = os.path.join(args.output_dir, 'watermarked')
    attacked_dir = os.path.join(args.output_dir, 'attacked')
    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(attacked_dir, exist_ok=True)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 构建模型
    H = W = 128  # 水印块大小
    default_noise_layer = ["Combined([Identity()])"]
    encoder_decoder = EncoderDecoder(H=H, W=W, message_length=message_length, 
                                    noise_layers=default_noise_layer)
    
    # 加载模型权重
    encoder_decoder.encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_best.pth')))
    encoder_decoder.decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder_best.pth')))
    encoder_decoder.to(device)
    encoder_decoder.encoder.eval()
    encoder_decoder.decoder.eval()
    print("模型已加载")

    
    # 搜索所有图像文件
    image_files = search_image_files(args.image_dir)
    print(f"找到 {len(image_files)} 张图像用于测试")
    
    # 开始测试
    results = {'res': [0, 0, 0, 0]}  # [PSNR, SSIM, BitAccuracy, CRCSuccess]
    total_images = len(image_files)
    
    for img_idx, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        try:
            # 读取原始图像
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # 生成随机水印信息
            if args.use_crc:
                message = np.random.choice([0, 1], (1, message_length - 8))
                message = crc(message, 'encode')  # 添加CRC校验
            else:
                message = np.random.choice([0, 1], (1, message_length))
            
            original_message = torch.Tensor(message).to(device)
            
            # 嵌入水印
            h_coor, w_coor, splitSize = generate_random_coor(img_tensor.shape[2], img_tensor.shape[3], 128)
            encoded_image = encode(
                encoder_decoder.encoder, 
                img_tensor, 
                message, 
                splitSize=splitSize,
                inputSize=H, 
                h_coor=h_coor, 
                w_coor=w_coor, 
                psnr=args.psnr_target
            )
            encoded_image = torch.clamp(encoded_image, -1, 1)
            
            # 保存水印图像
            img_name = os.path.basename(img_path)
            save_images((encoded_image + 1) / 2, os.path.join(watermarked_dir, img_name))
            
            # 计算图像质量
            psnr, ssim = image_quality_evaluate(img_tensor, encoded_image)
            results["res"][0] += psnr
            results["res"][1] += ssim
            
            # 无攻击下的解码
            decoded_message = decode(encoder_decoder.decoder, encoded_image)
            bit_accuracy = calc_bit_accuracy(original_message, decoded_message)
            results["res"][2] += bit_accuracy
            
            # CRC校验
            if args.use_crc:
                crc_success = crc_evaluate(decoded_message)
                results["res"][3] += crc_success
        
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue
    
    # 计算平均值
    for attack in results:
        for i in range(4):
            results[attack][i] /= total_images
    
    # 打印结果
    print("\n============= 测试结果 =============")
    print(f"测试图像数量: {total_images}")
    print(f"{'攻击类型':<15} {'PSNR':<10} {'SSIM':<10} {'Bit Accuracy':<15} {'CRC Success':<15}")
    for attack, metrics in results.items():
        print(f"{attack:<15} {metrics[0]:.4f}      {metrics[1]:.4f}      {metrics[2]:.4f}          {metrics[3]:.4f}")
    
    # 将结果保存到文件
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"测试图像数量: {total_images}\n")
        f.write(f"{'攻击类型':<15} {'PSNR':<10} {'SSIM':<10} {'Bit Accuracy':<15} {'CRC Success':<15}\n")
        for attack, metrics in results.items():
            f.write(f"{attack:<15} {metrics[0]:.4f}      {metrics[1]:.4f}      {metrics[2]:.4f}          {metrics[3]:.4f}\n")

if __name__ == '__main__':
    main()