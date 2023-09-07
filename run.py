import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from common.Fusionformer import Fusionformer
from common.refine import refine
from common.pretrain import pretrain
from common.camera import get_uvd2xyz
opt = opts().parse()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):


    loss_all = {'loss_post_refine': AccumLoss(),'loss_gt': AccumLoss(),'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model['trans'].train()
        model['refine'].train()
    else:
        model['trans'].eval()
        model['refine'].eval()  

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split =='train':
            output_3D = model['trans'](input_2D) 
        elif split =='test':
                input_2D, output_3D = input_augmentation(input_2D, model['trans'])

        N = input_2D.size(0)
        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0
        if split =='train':
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        elif split =='test':
            out_target_single = out_target
            gt_3D_single = gt_3D
 

        if opt.refine:  
            # """中间帧微调"""
            # out_target_single[:, :, 0] = 0
            # output_3D_single = output_3D[:, opt.pad].unsqueeze(1)
            # uvd = torch.cat((input_2D[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            # xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            # xyz[:, :, 0, :] = 0
            # post_out = model['refine'](output_3D_single, xyz)
            """所有帧微调"""
            uvd = torch.cat((input_2D[:, :, :, :], output_3D[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D, batch_cam)
            xyz[:, :, 0, :] = 0
            post_out = model['refine'](output_3D, xyz)
            if split == 'train':
                loss_post_refine = mpjpe_cal(post_out, out_target)

        else:
            loss_post_refine = torch.tensor([0.]).cuda()
        
        if split == 'train':
            loss_gt = mpjpe_cal(output_3D, out_target)
            loss = loss_gt + loss_post_refine
            N = input_2D.size(0)
            loss_all['loss_post_refine'].update(loss_post_refine.detach().cpu().numpy() * N, N)
            loss_all['loss_gt'].update(loss_gt.detach().cpu().numpy() * N, N)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            if not opt.refine:
                output_3D = output_3D[:, opt.pad].unsqueeze(1) 
            else:
                output_3D = post_out[:,opt.pad].unsqueeze(1)
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss_post_refine'].avg,loss_all['loss_gt'].avg,loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2
        

def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip     = model(input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


if __name__ == '__main__':
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    model = {}
    model['trans'] = nn.DataParallel(Fusionformer(num_frame=opt.frames, num_joints=opt.n_joints, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)).cuda()
    model['refine'] = nn.DataParallel(refine(opt)).cuda()

    trans_dict = model['trans'].state_dict()
    if opt.trans_reload:
        model_path = os.path.join(opt.previous_dir, opt.pretrained_trans)
        print(model_path)

        pre_dict_trans = torch.load(model_path)
        for name, key in trans_dict.items():
            trans_dict[name] = pre_dict_trans[name]
        model['trans'].load_state_dict(trans_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload:
        model_path = os.path.join(opt.previous_dir, opt.pretrained_refine)
        print(model_path)

        pre_dict_refine = torch.load(model_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)


    for epoch in range(1, opt.nepoch):
        if opt.train: 
            loss_post_refine,loss_gt,loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)

        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_trans_name = save_model_2(opt.previous_trans_name, opt.checkpoint, epoch, p1, model['trans'],'trans')
            if opt.refine:
                opt.previous_refine_name = save_model_2(opt.previous_refine_name, opt.checkpoint, epoch, p1, model['refine'],'refine')
            opt.previous_best_threshold = p1


        if opt.test:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        elif opt.train:
            logging.info('epoch: %d, lr: %.7f, loss_post_refine: %.4f, loss_gt: %.4f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss_post_refine,loss_gt,loss, p1, p2))
            print('e: %d, lr: %.7f, loss_post_refine: %.4f, loss_gt: %.4f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss_post_refine,loss_gt,loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)