import sys 
import os 
import os.path as osp 
import logging 
import time 

from tqdm import tqdm 

import utils





def eval(params, data_loader, model, criterion):
    model.eval()

    # ===== summary for current training loop 
    #       and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    acc_avg = utils.RunningAverage()


    end_time = time.time()

    loop = tqdm(enumerate(data_loader), total=len(data_loader)) # (ref) https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/VGG_for_CIFAR10.py
    for batch_idx, (inputs, targets) in loop:

        inputs = inputs.to(device=params.device)
        targets = targets.to(device=params.device)

        
        # ===== forward 
        scores = model(inputs)
        loss = criterion(scores, targets)  # batch_loss 
        acc = utils.calc_accuracy(scores, targets)


        # =====update the average loss, acc,  of the batch 
        loss_avg.update(loss.item())
        acc_avg.update(acc)


        # ===== (ref) https://github.com/tqdm/tqdm
        loop.set_postfix(acc=f'{acc_avg():05.3f}', loss=f'{loss_avg():05.3f}')
        loop.update

    
    logging.info(f"test_acc={acc_avg():05.3f}, avg_loss={loss_avg():05.3f}")

    return loss_avg(), acc_avg()