from torch_utils import *
import torch
from model import RDN
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim


def create_lr_image():
    data_path = 'data/DIV2K_valid_HR'
    scaling_factor = 8
    for file in os.listdir(data_path):
        # 加载图像
        img = Image.open(os.path.join(data_path, file), mode='r')
        hr_img = img.convert('RGB')
        # lr_img = hr_img.resize((int(hr_img.width / scaling_factor),
        #                         int(hr_img.height / scaling_factor)),
        #                        Image.BICUBIC)
        lr_img = hr_img.resize((2040, 1352), Image.BICUBIC)
        save_path = os.path.join('data/DIV2K_valid_HR_v2/' + file)
        lr_img.save(save_path)


def get_srgan_model(sf):
    # 模型参数
    n_features = 64  # 中间层通道数
    n_blocks = 16  # 残差模块数量
    n_channel = 3  # 输入通道
    # 预训练模型
    srgan_checkpoint = "save_model/checkpoint_srRDB2.pth"
    checkpoint = torch.load(srgan_checkpoint)
    generator = RDN(scale_factor=sf,
                    num_channels=n_channel,
                    num_features=n_features,
                    growth_rate=64,
                    num_blocks=n_blocks,
                    num_layers=8).to(device)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    return generator


def sr_test_method():
    lr_path = 'data/DIV2K_valid_LR_x4'
    hr_path = 'data/DIV2K_valid_HR'
    scaling_factor = 4  # 放大比例
    PSNR_list = []
    SSIM_list = []
    for lr_file, hr_file in zip(os.listdir(lr_path), os.listdir(hr_path)):
        # 测试图像
        lr_img_pt = os.path.join(lr_path, lr_file)
        hr_img_pt = os.path.join(hr_path, hr_file)
        # 加载图像
        lr_img = Image.open(lr_img_pt, mode='r')
        lr_img = lr_img.convert('RGB')
        hr_img = Image.open(hr_img_pt, mode='r')
        hr_img = hr_img.convert('RGB')
        # 双线性上采样
        # Bicubic_img = lr_img.resize((int(lr_img.width * scaling_factor), int(lr_img.height * scaling_factor)),
        #                             Image.BICUBIC)
        # Bicubic_img.save('results/prediction_img/bicubic_{}'.format(lr_file))
        # 图像预处理
        lr_img = convert_image(lr_img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)
        # 记录时间
        start = time.time()
        # 转移数据至设备
        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
        srgan_model = get_srgan_model(scaling_factor)
        # 模型推理
        with torch.no_grad():
            fake_img = srgan_model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            fake_img = convert_image(fake_img, source='[-1, 1]', target='pil')
            psnr = psnr1(np.array(fake_img), np.array(hr_img))
            single_ssim = ssim(np.array(fake_img), np.array(hr_img), multichannel=True)
            print(psnr)
            print(single_ssim)
            PSNR_list.append(psnr)
            SSIM_list.append(single_ssim)
            fake_img.save('results/prediction_img/srgan_{}'.format(lr_file))
        print('用时  {:.3f} 秒'.format(time.time() - start))
    print('平均SSIM:%.4f' % (sum(SSIM_list) / len(SSIM_list)))
    print('平均PSNR:%.4f' % (sum(PSNR_list) / len(PSNR_list)))


def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


if __name__ == '__main__':
    # create_lr_image()
    sr_test_method()
    # show_all_image()
