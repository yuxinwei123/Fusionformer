import argparse
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=512, type=int)
        self.parser.add_argument('--d_hid', default=1024, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        # self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', action='store_true')
        # self.parser.add_argument('--train', default='0',type=int)
        self.parser.add_argument('--pretrain', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=21)
        self.parser.add_argument('--batch_size', type=int, default=512, help='can be changed depending on your machine')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=0)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=9)
        # self.parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--vis_reload', type=int, default=1)
        self.parser.add_argument('--trans_reload', action='store_true')
        self.parser.add_argument('--refine_reload', action='store_true')
        self.parser.add_argument('--pretrain_reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='checkpoint/pretrained')
        self.parser.add_argument('--pretrained_trans', type=str, default='model_trans_4903.pth')
        self.parser.add_argument('--pretrained_refine', type=str, default='model_refine_4849.pth')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_trans_name', type=str, default='')
        self.parser.add_argument('-previous_refine_name', type=str, default='')
        self.parser.add_argument('-previous_pretrain_name', type=str, default='')
        self.parser.add_argument('--video', type=str, default='Walking 1.54138969.mp4', help='input video')
        # self.parser.add_argument('--gpu', type=str, default='0', help='input video')
    def parse(self):#返回命令行参数实例opt
        self.init()
        
        self.opt = self.parser.parse_args()

        if self.opt.test:#测试的过程中不进行训练
            self.opt.train = 0
            
        self.opt.pad = (self.opt.frames-1) // 2
        # self.opt.pad = 0
        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt





        
