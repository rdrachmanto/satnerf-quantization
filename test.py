import time

import rasterio
import torch

import sat_utils

# x = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
# print(x.to_sparse())
# torch.save(x, "check-size-non-sparse.pth")
# torch.save(x.to_sparse(), "check-size-sparse.pth")

# ins = torch.rand(size=[20])
# lin = torch.nn.Linear(in_features=20, out_features=50)
#
# print(ins)
# print(lin.weight.shape)
# print(lin(ins).shape)


# import torch
# from torchvision.models import resnet18
# import torch_pruning as tp
#
# model = resnet18(pretrained=True).eval()
#
# # 1. Build dependency graph for resnet18
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224))
#
# # 2. Group coupled layers for model.conv1
# group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])
#
# # 3. Prune grouped layers altogether
# if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
#     group.prune()
#
# # 4. Save & Load
# model.zero_grad()  # clear gradients
# torch.save(model, 'model.pth')  # We can not use .state_dict as the model structure is changed.
# model = torch.load('model.pth')  # load the pruned model
# print(model)

# ckpt = torch.load("/home/myid/zis35724/satnerf/pruning-result/saved-models/JAX_416_s-nerf_16layers_GroupNorm_1_all/ratio-0.9.pth", map_location="cpu")
# ckpt["coarse"].cuda()
# time.sleep(6000)


with rasterio.open("/data/datasets/Track3-preprocess-oma/OMA_042/ba/crops/OMA_042_035_RGB.tif", 'r') as f:
    img = f.read()
    print(img.shape[1] * img.shape[2])

d = sat_utils.read_dict_from_json("/data/datasets/Track3-preprocess-oma/OMA_042/ba/OMA_042_035_RGB.json")
print(d)



