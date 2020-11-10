import os
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable


from data import DataModule

from network import MixAttNet
from Utils import check_dir, AvgMeter, dice_score


def adjust_lr(optimizer, iteration, num_iteration):
    """
    we decay the learning rate by a factor of 0.1 in 1/2 and 3/4 of whole training process
    """
    if iteration == num_iteration // 2:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif iteration == num_iteration // 4 * 3:
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


def loss_func(predict, label, pos_weight):
    """
    here we define the loss function, which you can upload additional loss in here
    """
    bce_loss = F.binary_cross_entropy_with_logits(predict, label, pos_weight=pos_weight)
    return bce_loss


def train_batch(net, optimizer, loader, patch_size, batch_size):
    net.train()

    images, masks = next(iter(loader))

    # here we calculate the positive ratio in the input batch data
    if np.where(masks == 1)[0].shape[0] == 0:
        weight = 1
    else:
        weight = batch_size*patch_size[0]*patch_size[1]*patch_size[2]/np.where(masks == 1)[0].shape[0]

    images = Variable(images.cuda())
    masks  = Variable(masks.cuda())

    predict = net(images)

    optimizer.zero_grad()

    weight = torch.FloatTensor([weight]).cuda()
    loss1 = loss_func(predict[0], masks, pos_weight=weight)
    loss2 = loss_func(predict[1], masks, pos_weight=weight)
    loss3 = loss_func(predict[2], masks, pos_weight=weight)
    loss4 = loss_func(predict[3], masks, pos_weight=weight)
    loss5 = loss_func(predict[4], masks, pos_weight=weight)
    loss6 = loss_func(predict[5], masks, pos_weight=weight)
    loss7 = loss_func(predict[6], masks, pos_weight=weight)
    loss8 = loss_func(predict[7], masks, pos_weight=weight)
    loss9 = loss_func(predict[8], masks, pos_weight=weight)
    loss = loss1 + \
           0.8*loss2 + 0.7*loss3 + 0.6*loss4 + 0.5*loss5 + \
           0.8*loss6 + 0.7*loss7 + 0.6*loss8 + 0.5*loss9
    loss.backward()
    optimizer.step()
    return loss.item()


def val(net, loader):
    net.eval()
    metric_meter = AvgMeter()
    for batch in loader:
        images, masks = batch
        preds = net(images)
        metric_meter.update(dice_score(preds, masks))
    return metric_meter.avg


def main(args):
    torch.cuda.set_device(args.gpu_id)

    check_dir(args.output_path)
    ckpt_path = os.path.join(args.output_path, "ckpt")
    check_dir(ckpt_path)

    dm = DataModule(
            args.data_path,
            target_resolution=args.target_resolution,
            target_shape=args.target_shape,
            patch_size=args.patch_size,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers
        )
    dm.setup()
    train_dataloader, val_dataloder = dm.train_dataloader(),  dm.val_dataloader()


    net = MixAttNet().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    open(os.path.join(args.output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    for iteration in range(1, args.num_iteration+1):
        adjust_lr(optimizer, iteration, args.num_iteration)
        train_loss = train_batch(net=net, optimizer=optimizer, loader=train_dataloader, 
                                 patch_size=args.patch_size, batch_size=args.train_batch_size)
        loss_meter.update(train_loss)

        if iteration % args.pre_fre == 0:
            iteration_time = time.time() - start_time
            info = [iteration, loss_meter.avg, iteration_time]
            print("Iter[{}] | Loss: {:.3f} | Time: {:.2f}".format(*info))
            start_time = time.time()
            loss_meter.reset()

        if iteration % args.val_fre == 0:
            val_dice = val(net, dm.val_dataloader())
            if val_dice > best_metric:
                torch.save(net.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(args.output_path, "train_record.txt"), 'a+').write("{:.3f} | {:.3f}\n".format(train_loss, val_dice))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
        if iteration % 100 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))


if __name__ == '__main__':

    class Parser(object):
        def __init__(self):
            self.gpu_id = 0

            self.target_resolution = (1.5, 1.5, 8.),
            self.target_shape = None,

            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.train_batch_size = 4
            self.val_batch_size = 4
            self.num_workers = 4

            self.num_iteration = 4000
            self.val_fre = 200
            self.pre_fre = 20

            self.patch_size = (64, 64, 64)

            self.data_path = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train"
            self.output_path = "output/"

    parser = Parser()
    main(parser)
