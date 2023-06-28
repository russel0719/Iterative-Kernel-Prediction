import time
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.plot_image import plot
from utils.metrics import SSIM, PSNR
from utils.utils import tensor2imgtensor, imgtensor2img

def test(test_dl, len_test, predictor, sftmd, args):
    loss_fn = F.mse_loss
    num_iter = args['num_iter']
    device = args['device']
    scale = args['scale']
    
    predictor.to(device)
    sftmd.to(device)
    predictor.eval()
    sftmd.eval()
    count = 0
    
    total_mse = 0
    total_ssim = 0
    total_psnr = 0
    total_time = 0
    
    with torch.no_grad():
        for test_data in test_dl:
            LR = test_data['LR']
            HR = test_data['HR']
            
            n_batch = len(LR)
            count += n_batch
            
            LR = LR.to(device)
            HR = HR.to(device)
            
            start_time = time.time()
            
            estim_kernel = torch.empty(
                (LR.size()[0], args['reduced_kernel_dim']),
                dtype=torch.float32, requires_grad=False
            )
            nn.init.normal_(estim_kernel)
            estim_kernel = estim_kernel.to(device)
            
            for i in range(num_iter):
                SR = sftmd(LR, estim_kernel)
                estim_kernel = predictor(SR)
                
            end_time = time.time()

            mse = loss_fn(SR, HR)
            ssim = SSIM(SR, HR)
            psnr = PSNR(SR, HR)
            total_mse += mse
            total_ssim += ssim
            total_psnr += psnr
            total_time += end_time - start_time

            msg = "{}\t[{}/{}]\ttest [mse: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                time.ctime(), count, len_test, mse, ssim, psnr
            )
            print(msg)
            
            result_lr = tensor2imgtensor(LR[0].cpu())
            result_sr = tensor2imgtensor(SR[0].cpu())
            result_hr = tensor2imgtensor(HR[0].cpu())
            
            img = imgtensor2img(result_sr)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{args['result_dir']}/{args['test'][12:]}_{count}_x{scale}.jpg", img)
            
            idx_from, idx_to = test_data['patch_from'][0], test_data['patch_to'][0]
            patch_lr = result_lr[idx_from[0]:idx_to[0], idx_from[1]:idx_to[1], :]
            patch_sr = result_sr[idx_from[0]*scale:idx_to[0]*scale, idx_from[1]*scale:idx_to[1]*scale, :]
            patch_hr = result_hr[idx_from[0]*scale:idx_to[0]*scale, idx_from[1]*scale:idx_to[1]*scale, :]
            plot([patch_lr, patch_sr, patch_hr], ["LR", "SR", "HR"], (1, 3))
    
    print(
        "avg mse: {:.5f}\navg ssim: {:.5f}\navg psnr: {:.5f}\navg time: {:.5f} sec".format(
            total_mse/len_test, total_ssim/len_test, total_psnr/len_test, total_time/len_test
        )
    )