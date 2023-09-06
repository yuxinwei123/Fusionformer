import math
import logging
from functools import partial
from collections import OrderedDict
from operator import concat
from re import X
from statistics import mode
from symbol import parameters
from turtle import forward
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from thop import profile

from VIT import *

class CTA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_mask = nn.Parameter(torch.ones(1, 1, 1))
        self.k_mask = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q_mask = self.q_mask.expand(*q.size())
        k_mask = self.k_mask.expand(*k.size())

        q = q * q_mask
        k = k * k_mask

        q_mask = (q == 0).bool()
        k_mask = (k == 0).bool()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CTA_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CTA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Spatial_Encoder(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        
    def forward(self, x):
        B, T, P, C = x.shape  ##### B is batch size, T is number of frames, P is number of joints, C is Channel of joints
        x = rearrange(x, 'B T P C  -> (B T) P C')

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(B T) P D -> B T P D', T=T)#### D is embedding dim
        return x

class Temporal_Encoder(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame,embed_dim_ratio*num_joints))
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio*num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Temporal_norm = norm_layer(embed_dim_ratio*num_joints)
    def forward(self,x):
        B,T,P,D = x.shape
        x = rearrange(x, 'B T P D -> B T (P D)', T=T)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.Temporal_blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        x = rearrange(x, 'B T (P D) -> B T P D', T=T,P=P,D=D)
        return x

class Self_Trajectory_Encoder(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Self_Trajectory_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Self_Trajectory_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio))
        self.Self_Trajectory_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Self_Trajectory_norm = norm_layer(embed_dim_ratio)
    def forward(self,x):
        B,T,P,C= x.shape
        x = rearrange(x, 'B T P C -> (B P) T C', T=T)
        x = self.Self_Trajectory_patch_to_embedding(x)
        x += self.Self_Trajectory_pos_embed
        x = self.pos_drop(x)
        for blk in self.Self_Trajectory_blocks:
            x = blk(x)
        x = self.Self_Trajectory_norm(x)
        x = rearrange(x, '(B P) T D -> B T P D', B=B,P=P)
        return x

class Cross_Trajectory_Encoder(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Cross_Trajectory_pos_embed = nn.Parameter(torch.zeros(1, num_joints, num_frame*embed_dim_ratio))
        self.Cross_Trajectory_blocks = nn.ModuleList([
            CTA_Block(
                dim=num_frame*embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Cross_Trajectory_norm = norm_layer(num_frame*embed_dim_ratio)
    def forward(self,x):
        B,T,P,D= x.shape
        x = rearrange(x, 'B T P D -> B P (T D)', T=T)
        x += self.Cross_Trajectory_pos_embed
        x = self.pos_drop(x)
        for blk in self.Cross_Trajectory_blocks:
            x = blk(x)
        x = self.Cross_Trajectory_norm(x)
        x = rearrange(x, 'B P (T D) -> B T P D', B=B,P=P,T=T)
        return x

class Global_local_fusion_block(nn.Module):
    def __init__(self,num_frame,embed_dim_ratio,num_joints,channel,reducion=16):
        super().__init__()
        self.num_frame = num_frame
        self.Global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reducion,bias=False),
            nn.ReLU(inplace=True),#inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.Linear(channel//reducion,channel,bias=False),
            nn.Sigmoid()
            )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio * num_joints),
            nn.Conv1d(4*num_frame,num_frame,kernel_size=1)
        )
    def forward(self,x):
        B,T,P,D= x.shape
        y = self.Global_pooling(x).view(B,T)
        y = self.fc(y).view(B,T,1,1)
        y = x*y.expand_as(x)
        y = rearrange(y, 'B T P D -> B T (P D)', B=B,P=P,T=T)
        y = self.fusion(y)
        y = rearrange(y, 'B T (P D) -> B T P D', B=B,P=P,T=self.num_frame)
        return y

class Fusionformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,depth1=4,depth2=4,depth3=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        super().__init__()
        self.SE = Spatial_Encoder(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,drop_path_rate=drop_path_rate)
        self.TE = Temporal_Encoder(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,drop_path_rate=drop_path_rate)
        self.STE = Self_Trajectory_Encoder(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,drop_path_rate=drop_path_rate)
        self.CTE = Cross_Trajectory_Encoder(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,drop_path_rate=drop_path_rate)

        self.fcn = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio * num_joints),
            nn.Conv1d(4*num_frame,num_frame,kernel_size=1)
        )
        self.glf = Global_local_fusion_block(num_frame=num_frame,embed_dim_ratio=embed_dim_ratio,num_joints=num_joints,channel=4*num_frame,reducion=4*num_frame)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio , 3),
        )
    def forward(self,x):
        B, T, P, C = x.shape
        #GIM
        SE_feature = self.SE(x)
        TE_feature = self.TE(SE_feature)
        x_G = torch.cat([SE_feature,TE_feature],dim=1)#Concat along T
        #LIM
        STE_feature = self.STE(x)
        CTE_feature = self.CTE(STE_feature)
        x_L = torch.cat([STE_feature ,CTE_feature],dim=1)#Concat along T
        #glf
        Concat_feature = torch.cat([x_G,x_L],dim=1)#Concat along T
        fusion_feature = self.glf(Concat_feature)
        out = self.head(fusion_feature)
        return out



