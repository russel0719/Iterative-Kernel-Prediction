import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.metrics import SSIM, PSNR

def train(train_dl, valid_dl, len_train, predictor, sftmd, args):
    optimizer_p = optim.Adam(predictor.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']))
    optimizer_f = optim.Adam(sftmd.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']))
    scheduler_p = lr_scheduler.StepLR(optimizer_p, step_size=args['step_size'], gamma=args['gamma'])
    scheduler_f = lr_scheduler.StepLR(optimizer_f, step_size=args['step_size'], gamma=args['gamma'])
    loss_fn = F.mse_loss
    num_epoch = args['num_epoch']
    num_iter = args['num_iter']
    device = args['device']
    log_interval = args['log_interval']
    
    ckpt_model_path_p = ''
    ckpt_model_path_f = ''
    best_val_p_loss = math.inf
    best_val_f_loss = math.inf
    
    for epoch in range(num_epoch):
        sftmd.to(device)
        sftmd.train()
        predictor.to(device)
        predictor.train()
        train_p_loss = 0
        train_f_loss = 0
        count = 0
        
        for batch_id, train_data in enumerate(train_dl):
            LR = train_data['LR']
            HR = train_data['HR']
            reduced_kernel = train_data['reduced_kernel']
            
            n_batch = len(LR)
            count += n_batch
            
            LR = LR.to(device)
            HR = HR.to(device)
            reduced_kernel = reduced_kernel.to(device)
            
            estim_kernel = torch.empty(reduced_kernel.size(), dtype=torch.float32, requires_grad=False)
            nn.init.normal_(estim_kernel)
            estim_kernel = estim_kernel.to(device)
            
            # IKC Algorithm
            for i in range(num_iter):
                SR = sftmd(LR, estim_kernel.detach())
                loss_f = loss_fn(SR, HR)
                train_f_loss += loss_f
                loss_f.backward()
                optimizer_f.step()
                optimizer_f.zero_grad()
                
                estim_kernel = predictor(SR.detach())
                loss_p = loss_fn(estim_kernel, reduced_kernel)
                train_p_loss += loss_p
                loss_p.backward()
                optimizer_p.step()
                optimizer_p.zero_grad()

            if (batch_id + 1) % log_interval == 0:
                msg = "{}\tEpoch {}:[{}/{}]\ttrain [predictor: {:.5f}  sftmd: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                    time.ctime(), epoch + 1, count, len_train,
                    train_p_loss / (batch_id + 1), train_f_loss / (batch_id + 1) / num_iter,
                    SSIM(SR, HR), PSNR(SR, HR)
                )
                print(msg)
        scheduler_p.step()
        scheduler_f.step()
        
        with torch.no_grad():
            predictor.eval()
            sftmd.eval()
            valid_p_loss = 0
            valid_f_loss = 0
            ssim = 0
            psnr = 0
            for valid_data in valid_dl:
                LR = valid_data['LR']
                HR = valid_data['HR']
                reduced_kernel = valid_data['reduced_kernel']

                n_batch = len(LR)
                LR = LR.to(device)
                HR = HR.to(device)
                reduced_kernel = reduced_kernel.to(device)
            
                estim_kernel = torch.empty(reduced_kernel.size(), dtype=torch.float32, requires_grad=False)
                nn.init.normal_(estim_kernel)
                estim_kernel = estim_kernel.to(device)

                # IKC Algorithm
                for i in range(num_iter):
                    SR = sftmd(LR, estim_kernel)
                    loss_f = loss_fn(SR, HR)
                    valid_f_loss += loss_f

                    estim_kernel = predictor(SR)
                    loss_p = loss_fn(estim_kernel, reduced_kernel)
                    valid_p_loss += loss_p
                ssim += SSIM(SR, HR)
                psnr += PSNR(SR, HR)
            msg = "{}\tEpoch {}:\t\tvalid [predictor: {:.5f}  sftmd: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                time.ctime(), epoch + 1, valid_p_loss / len(valid_dl), valid_f_loss / len(valid_dl) / num_iter,
                ssim / len(valid_dl), psnr / len(valid_dl)
            )
            print(msg)

            # save best model
            if (valid_p_loss / len(valid_dl) < best_val_p_loss):
                best_val_p_loss = valid_p_loss / len(valid_dl)
                print("predictor : new best validation loss!")
                predictor.eval().cpu()
                ckpt_model_filename_p = "ckpt_predictor_x" + str(args['scale']) + ".pth"
                ckpt_model_path_p = os.path.join(args['ckpt_dir'], ckpt_model_filename_p)
                torch.save({
                    'model_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer_p.state_dict(),
                    'total_loss': best_val_p_loss
                }, ckpt_model_path_p)
                
            if (valid_f_loss / len(valid_dl) < best_val_f_loss):
                best_val_f_loss = valid_f_loss / len(valid_dl)
                print("sftmd : new best validation loss!")
                sftmd.eval().cpu()
                ckpt_model_filename_f = "ckpt_sftmd_x" + str(args['scale']) + ".pth"
                ckpt_model_path_f = os.path.join(args['ckpt_dir'], ckpt_model_filename_f)
                torch.save({
                    'model_state_dict': sftmd.state_dict(),
                    'optimizer_state_dict': optimizer_f.state_dict(),
                    'total_loss': best_val_f_loss
                }, ckpt_model_path_f)