## Fusionformer : Exploiting the Joint Motion Synergy with Fusion Network Based On Transformer for 3D Human Pose Estimation

### Dataset

Our code is compatible with the dataset setup introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset (./dataset directory).



### Evaluating pre-trained models

We provide the pre-trained 9-frame model (CPN detected 2D pose as input) . To evaluate it, put it into the `./checkpoint/pretrained` directory and run:

`python run.py --test --trans_reload --pretrained_trans model_trans_4903.pth`

We provide the pre-trained 9-frame  refinement model (CPN detected 2D pose as input). To evaluate it, put it into the `./checkpoint/pretrained` directory and run:

`python run.py --test --trans_reload --pretrained_trans model_trans_4903.pth --refine_reload --pretrained_refine model_refine_4849.pth`



### Training new models

- To train a model from scratch (CPN detected 2D pose as input), run:

  `python run.py --train -k cpn_ft_h36m_dbb --lr 1e-3`

- To train a model from scratch (Ground truth 2D pose as input), run:

  `python run.py --train -k gt --lr 1e-3`

  

### Visualization and other functions

We keep our code consistent with [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to their project page for further information.