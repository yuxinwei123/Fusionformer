import torch
import torch.nn as nn
from common.opt import opts
opt = opts().parse()
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class pretrain(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_features = opt.in_channels*opt.n_joints*opt.frames
        hidden_features = 1024
        out_features = opt.in_channels*opt.n_joints*opt.frames
        self.post_refine = Mlp(in_features,hidden_features,out_features)

    def forward(self,x):
        B,F,P,_ = x.size()
        x = x.view(B, -1)
        x = self.post_refine(x).view(B,F,P,2) 

        return x

if __name__ == '__main__':
    t = torch.rand(512,3,17,2)
    model = pretrain(opt)
    r = model(t)
    print(r.shape)


