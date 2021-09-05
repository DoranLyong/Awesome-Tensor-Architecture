# (ref) https://github.com/DoranLyong/Awesome-Human-Activity-Recognition/blob/main/tutorial_04/train.py

import sys 
import os 
import os.path as osp 
import logging 
import time 

from tqdm import tqdm 

import utils



def train_epoch(params, epoch:int, data_loader, model, criterion, optimizer, epoch_logger, batch_logger):
    
    model.train()
    

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

                


        # ==== backward
        #      clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()


        # ==== gradient descent, and step
        optimizer.step()



        # ===== update the average loss, acc,  of the batch 
        loss_avg.update(loss.item())
        acc_avg.update(acc)



        # ===== Print & Save logs 
        batch_logger.log({  'epoch': epoch,
                            'batch': batch_idx + 1,
                            'iter': (epoch - 1) * len(data_loader) + (batch_idx + 1),
                            'loss': loss_avg(),
                            'acc': acc_avg(),
                            'lr': optimizer.param_groups[0]['lr'], }
                        )  



        # ===== (ref) https://github.com/tqdm/tqdm
        loop.set_postfix(acc=f'{acc_avg():05.3f}', loss=f'{loss_avg():05.3f}', lr=optimizer.param_groups[0]['lr'])
        loop.update


    # ===== Epoch log     
    logging.info(f"Traing epoch @ {epoch + 1}/{params.num_epochs}, acc={acc_avg():05.3f}, avg_loss={loss_avg():05.3f},  lr={optimizer.param_groups[0]['lr']}")

    epoch_logger.log({  'epoch': epoch,
                        'loss': loss_avg(),
                        'acc': acc_avg(),
                        'lr': optimizer.param_groups[0]['lr'],}
                    )




        