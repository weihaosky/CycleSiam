import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter

from datasets.siam_cycle_dataset import DataSets

from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamMask in json format')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')


best_acc = 0.

def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.propagate = False
    logger.info("\n" + collect_env_info())
    logger.info(args)

    cfg = load_config(args)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    if args.log_dir:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = Dummy()

    # build dataset
    train_loader, val_loader = build_data_loader(cfg)

    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(pretrain=True, anchors=cfg['anchors'])
    else:
        exit()
    logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    model = model.cuda()
    dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)

    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_acc, arch = restore_from(model, optimizer, args.resume)
        dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    logger.info(lr_scheduler)

    logger.info('model prepare done')

    train(train_loader, dist_model, optimizer, lr_scheduler, args.start_epoch, cfg)


def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):
    global tb_index, best_acc, cur_lr, logger
    cur_lr = lr_scheduler.get_cur_lr()
    logger = logging.getLogger('global')
    avg = AverageMeter()
    model.train()
    model = model.cuda()
    end = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    start_epoch = epoch
    epoch = epoch
    for iter, input in enumerate(train_loader):

        if epoch != iter // num_per_epoch + start_epoch:  # next epoch
            epoch = iter // num_per_epoch + start_epoch

            if not os.path.exists(args.save_dir):  # makedir/save model
                os.makedirs(args.save_dir)

            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))

            if epoch == args.epochs:
                return

            if model.module.features.unfix(epoch/args.epochs):
                logger.info('unfix part model.')
                optimizer, lr_scheduler = build_opt_lr(model.module, cfg, args, epoch)

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()

            logger.info('epoch:{}'.format(epoch))

        tb_index = iter + num_per_epoch * start_epoch
        if iter % num_per_epoch == 0 and iter != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                tb_writer.add_scalar('lr/group%d' % (idx+1), pg['lr'], tb_index)

        data_time = time.time() - end
        avg.update(data_time=data_time)
        track12 = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0][0]).cuda(),
            'search': torch.autograd.Variable(input[0][1]).cuda(),
            'label_cls': torch.autograd.Variable(input[0][2]).cuda(),
            'label_loc': torch.autograd.Variable(input[0][3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[0][4]).cuda(),
            'template_bbox': torch.autograd.Variable(input[0][5]).cuda(),
            'label_mask': torch.autograd.Variable(input[0][6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[0][7]).cuda(),
        }
        track21 = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[1][0]).cuda(),
            'search': torch.autograd.Variable(input[1][1]).cuda(),
            'label_cls': torch.autograd.Variable(input[1][2]).cuda(),
            'label_loc': torch.autograd.Variable(input[1][3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[1][4]).cuda(),
            'template_bbox': torch.autograd.Variable(input[1][5]).cuda(),
            'label_mask': torch.autograd.Variable(input[1][6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[1][7]).cuda(),
        }
        track23 = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[2][0]).cuda(),
            'search': torch.autograd.Variable(input[2][1]).cuda(),
            'label_cls': torch.autograd.Variable(input[2][2]).cuda(),
            'label_loc': torch.autograd.Variable(input[2][3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[2][4]).cuda(),
            'template_bbox': torch.autograd.Variable(input[2][5]).cuda(),
            'label_mask': torch.autograd.Variable(input[2][6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[2][7]).cuda(),
        }
        track32 = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[3][0]).cuda(),
            'search': torch.autograd.Variable(input[3][1]).cuda(),
            'label_cls': torch.autograd.Variable(input[3][2]).cuda(),
            'label_loc': torch.autograd.Variable(input[3][3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[3][4]).cuda(),
            'template_bbox': torch.autograd.Variable(input[3][5]).cuda(),
            'label_mask': torch.autograd.Variable(input[3][6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[3][7]).cuda(),
        }

        # ========================== cycle forward frame1 -> frame2 ===================================
        outputs12 = model(track12, softmax=False)
        out_patch12 = trackres(cfg, outputs12, track12)
        track23['template'] = torch.autograd.Variable(torch.from_numpy(out_patch12).float()).cuda()

        # ========================== cycle forward frame2 -> frame3 ===================================
        outputs23 = model(track23, softmax=False)
        out_patch23 = trackres(cfg, outputs23, track23)
        track32['template'] = torch.autograd.Variable(torch.from_numpy(out_patch23).float()).cuda()

        # ========================== cycle backward frame3 -> frame2 ===================================
        outputs32 = model(track32, softmax=False)
        out_patch32 = trackres(cfg, outputs32, track32)
        track21['template'] = torch.autograd.Variable(torch.from_numpy(out_patch32).float()).cuda()

        # ========================== cycle backward frame2 -> frame1 ===================================
        outputs = model(track21, softmax=True)

        rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = torch.mean(outputs['losses'][0]), torch.mean(outputs['losses'][1]), torch.mean(outputs['losses'][2])
        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])

        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']

        # CycleSiam: training with only bbox
        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight
        # CycleSiam+: training with bbox and mask
        # loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight

        optimizer.zero_grad()
        loss.backward()

        if cfg['clip']['split']:
            torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
            torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
            torch.nn.utils.clip_grad_norm_(model.module.mask_model.parameters(), cfg['clip']['mask'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        siammask_loss = loss.item()

        batch_time = time.time() - end

        rpn_cls_loss=min(rpn_cls_loss.item(), 1)
        rpn_loc_loss=min(rpn_loc_loss.item(), 1)
        rpn_mask_loss=min(rpn_mask_loss.item(), 1)
        mask_iou_mean=mask_iou_mean.item()
        mask_iou_at_5=mask_iou_at_5.item()
        mask_iou_at_7=mask_iou_at_7.item()

        avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                   rpn_mask_loss=rpn_mask_loss, siammask_loss=siammask_loss,
                   mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)

        tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
        tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)
        tb_writer.add_scalar('loss/mask', rpn_mask_loss, tb_index)
        tb_writer.add_scalar('mask/mIoU', mask_iou_mean, tb_index)
        tb_writer.add_scalar('mask/AP@.5', mask_iou_at_5, tb_index)
        tb_writer.add_scalar('mask/AP@.7', mask_iou_at_7, tb_index)
        end = time.time()

        if (iter + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                        '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}\t{rpn_mask_loss:s}\t{siammask_loss:s}'
                        '\t{mask_iou_mean:s}\t{mask_iou_at_5:s}\t{mask_iou_at_7:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                        data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                        rpn_mask_loss=avg.rpn_mask_loss, siammask_loss=avg.siammask_loss, mask_iou_mean=avg.mask_iou_mean,
                        mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
            print_speed(num_per_epoch * start_epoch + iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


def trackres(cfg, outputs12, track12):
    delta = outputs12['predict'][0]        
    score = outputs12['predict'][1]
    delta = delta.contiguous().view(delta.shape[0], 4, -1).data.cpu().numpy()
    score = F.softmax(score.contiguous().view(score.shape[0], 2, -1), dim=1).data[:, 1].cpu().numpy()

    anchor = generate_anchor(cfg['anchors'], 25)  # anchors: anchor cfg
    delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
    delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
    delta[:, 2, :] = np.exp(delta[:, 2, :]) * anchor[:, 2]
    delta[:, 3, :] = np.exp(delta[:, 3, :]) * anchor[:, 3]

    p = TrackerConfig()
    
    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[:,0] + wh[:,1]) * 0.5
        sz2 = (wh[:,0] + pad) * (wh[:,1] + pad)
        return np.sqrt(sz2)

    target_sz = np.array(track12['template_bbox'].cpu().numpy()[:,2:])
    scale_x = np.ones(target_sz.shape[0])

    # size penalty
    target_sz_in_crop = target_sz*scale_x[:,None]
    s_c = change(sz(delta[:, 2, :], delta[:, 3, :]) / (sz_wh(target_sz_in_crop))[:,None])  # scale penalty
    r_c = change((target_sz_in_crop[:,0] / target_sz_in_crop[:,1])[:,None] / (delta[:, 2, :] / delta[:, 3, :]))  # ratio penalty
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)
    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence

    best_pscore_id = np.argmax(pscore, 1)
    pred_in_crop = delta[range(best_pscore_id.shape[0]), :, best_pscore_id] / scale_x[:,None]
    # lr = penalty[range(best_pscore_id.shape[0]),best_pscore_id] * score[range(best_pscore_id.shape[0]),best_pscore_id] * p.lr  # lr for OTB

    res_cx = pred_in_crop[:, 0] + (track12['search'].shape[2] + 1) // 2
    res_cy = pred_in_crop[:, 1] + (track12['search'].shape[3] + 1) // 2
    res_w = pred_in_crop[:, 2]
    res_h = pred_in_crop[:, 3]
    target_pos = np.array([res_cx, res_cy]).T
    target_sz = np.array([res_w, res_h]).T
    
    def draw(image, box, name):
        image = np.transpose(image, (1,2,0)).copy()
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        image=cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.imwrite(name, image)

    debug = False
    if debug:
        batch_id = 0
        img_search = track12['search'][batch_id].cpu().numpy()
        box = [res_cx[batch_id] - res_w[batch_id] / 2., res_cy[batch_id] - res_h[batch_id] / 2., 
                res_cx[batch_id] + res_w[batch_id] / 2., res_cy[batch_id] + res_h[batch_id] / 2.]
        draw(img_search, box, "debug/{:06d}_pred.jpg".format(iter))
        img_temp = track12['template'][batch_id].cpu().numpy()
        box_temp = track12['template_bbox'][batch_id].cpu().numpy()
        draw(img_temp, box_temp, "debug/{:06d}_temp.jpg".format(iter))

    im_sz = track12['search'].shape[-2:]
    avg_chans = np.mean(track12['search'].cpu().numpy(), axis=(2, 3))
    sz = p.exemplar_size
    c = (sz + 1) / 2
    context_xmin = (target_pos[:, 0] - c).round()
    context_xmax = context_xmin + sz - 1
    context_ymin = (target_pos[:, 1] - c).round()
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin.min()))
    top_pad = int(max(0., -context_ymin.min()))
    right_pad = int(max(0., context_xmax.max() - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax.max() - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    im = track12['search'].cpu().numpy()
    k, r, c = im.shape[-3:]
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((args.batch, k, r + top_pad + bottom_pad, c + left_pad + right_pad), np.uint8)
        te_im[:, :, top_pad:top_pad + r, left_pad:left_pad + c] = im
        if top_pad:
            te_im[:, :, 0:top_pad, left_pad:left_pad + c] = avg_chans[:, :, None, None]
        if bottom_pad:
            te_im[:, :, r + top_pad:, left_pad:left_pad + c] = avg_chans[:, :, None, None]
        if left_pad:
            te_im[:, :, :, 0:left_pad] = avg_chans[:, :, None, None]
        if right_pad:
            te_im[:, :, :, c + left_pad:] = avg_chans[:, :, None, None]
        im_patch_original = np.zeros((args.batch, k, sz, sz), np.uint8)
        for id in range(args.batch):
            im_patch_original[id] = te_im[id, :, int(context_ymin[id]):int(context_ymax[id] + 1), int(context_xmin[id]):int(context_xmax[id] + 1)]
    else:
        im_patch_original = np.zeros((args.batch, k, sz, sz), np.uint8)
        for id in range(args.batch):
            im_patch_original[id] = im[id, :, int(context_ymin[id]):int(context_ymax[id] + 1), int(context_xmin[id]):int(context_xmax[id] + 1)]
    im_patch = im_patch_original

    return im_patch


if __name__ == '__main__':
    main()
