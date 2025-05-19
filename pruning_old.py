import time
import torch
import os
import json
from datasets import SatelliteDataset
import metrics
import numpy as np
import sat_utils
import argparse
import glob
import shutil
from eval_satnerf import load_nerf, batched_inference, predefined_val_ts, save_nerf_output_to_images
from torch.nn.utils import prune
from torch.nn.utils.prune import random_unstructured, l1_unstructured, global_unstructured, \
    random_structured, ln_structured


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def count_nonzero_parameters(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())


def eval_aoi(run_id, logs_dir, output_dir, epoch_number, split="val", checkpoints_dir=None, root_dir=None, img_dir=None, gt_dir=None, prune_amount=0.0):
    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # if gt_dir is not None:
    #     assert os.path.isdir(gt_dir)
    #     args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir

    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)
    parameters = [models["coarse"].fc_net[i] for i in range(0, 16, 2)]

    # Unstructured pruning
    for i in range(0, 16, 2):
        # random_unstructured(models["coarse"].fc_net[i], "weight", 0.5)
        l1_unstructured(models["coarse"].fc_net[i], "weight", 0.5)
    global_unstructured([(p, "weight") for p in parameters], pruning_method=prune.L1Unstructured, amount=prune_amount)

    # Structured pruning
    for i in range(0, 16, 2):
        random_structured(models["coarse"].fc_net[i], amount=prune_amount, dim=0)

    # Remove pruning masks
    for p in parameters:
        prune.remove(p, "weight")
        # print(type(p.weight))
        # print(p.weight)
        # for pp in p.parameters():
        #     pp.weight = torch.nn.Parameter(pp.to_sparse())
            # print(p.weight.values().shape)
            # print(p.detach().shape)
            # print(type(p.to_sparse()))

    for p in parameters:
        print(p.weight.shape)

    all_count = 0
    for p in parameters:
        all_count += count_nonzero_parameters(p)
    print("Params count: ", all_count)
    print("Params compression: ", 462336/all_count)

    # torch.save(parameters[:1], f"exp-pruning/params-prune-{prune_amount}-non-sparse-1L.pth")
    # torch.save(parameters[:2], f"exp-pruning/params-prune-{prune_amount}-non-sparse-2L.pth")
    # torch.save(parameters[:3], f"exp-pruning/params-prune-{prune_amount}-non-sparse-3L.pth")
    # torch.save(parameters[:4], f"exp-pruning/params-prune-{prune_amount}-non-sparse-4L.pth")
    # torch.save(parameters[:5], f"exp-pruning/params-prune-{prune_amount}-non-sparse-5L.pth")
    # torch.save(parameters[:6], f"exp-pruning/params-prune-{prune_amount}-non-sparse-6L.pth")
    # torch.save(parameters[:7], f"exp-pruning/params-prune-{prune_amount}-non-sparse-7L.pth")
    # torch.save(parameters[:8], f"exp-pruning/params-prune-{prune_amount}-non-sparse-8L.pth")
    # torch.save(parameters[:8], f"exp-pruning/orig-params-8L.pth")
    # torch.save(models["coarse"].state_dict(), f"exp-pruning/{run_id}-prune-{prune_amount}.ckpt")

    # prepare dataset
    dataset = SatelliteDataset(args.root_dir,
                               args.img_dir,
                               split="val",
                               img_downscale=args.img_downscale,
                               cache_dir=args.cache_dir
    )

    if split == "train":
        with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(1, len(dataset))

    psnr, ssim, mae = [], [], []

    for i in samples_to_eval:

        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id  = sample["src_id"]
        if "h" in sample and "w" in sample:
            W, H = sample["w"], sample["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

        ts = None
        if args.model == "sat-nerf":
            if split == "val":
                t = predefined_val_ts(src_id)
                ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
            else:
                ts = sample["ts"].cuda().squeeze()

        start = time.time()
        results = batched_inference(models, rays, ts, args)
        print("Inference time: ", time.time() - start)

        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        out_dir = os.path.join(output_dir, run_id, split)
        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

        # image metrics
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu())
        psnr.append(psnr_)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W).cpu(), rgbs.view(1, 3, H, W).cpu())
        ssim.append(ssim_)

        # geometry metrics
        pred_dsm_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        mae_ = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, args.gt_dir, out_dir, epoch_number)
        mae.append(mae_)
        print("{}: pnsr {:.3f} / ssim {:.3f} / mae {:.3f}".format(src_id, psnr_, ssim_, mae_))

        # clean files
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_diff_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm_diff"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)

    print("\nMean PSNR: {:.3f}".format(np.mean(np.array(psnr))))
    print("Mean SSIM: {:.3f}".format(np.mean(np.array(ssim))))
    print("Mean MAE: {:.3f}\n".format(np.mean(np.array(mae))))
    # return np.mean(np.array(psnr)), np.mean(np.array(ssim)), np.mean(np.array(mae))

scene = "JAX_260"
root_dir = f"/data/datasets/Track3-preprocess/{scene}/ba"
img_dir = f"/data/datasets/Track3-preprocess/{scene}/ba/crops"
gt_dir = f"/data/datasets/Track3-Truth-JAX"
cache_dir = f"/data/datasets/Track3-preprocess/{scene}/ba/cache"
output_dir = "exp-pruning"
base_dir = "/data/satnerf-exp/exp-batch"

# project_dir = "JAX_070_ds1_2gpu_batch16384_nerf"
# run_id = "2023-11-05_13-47-31_JAX_070_ds1_2gpu_batch16384_nerf"
# epoch_number = 20

# project_dir = "JAX_070_ds1_2gpu_batch1024_s-nerf"
# run_id = "2023-11-22_18-41-17_JAX_070_ds1_2gpu_batch1024_s-nerf"
# epoch_number = 4

project_dir = "JAX_070_ds1_2gpu_batch1024_sat-nerf"
run_id = "2023-11-22_15-41-51_JAX_070_ds1_2gpu_batch1024_sat-nerf"
epoch_number = 12

# project_dir = "JAX_260_ds1_2gpu_batch1024_s-nerf"
# run_id = "2023-12-04_17-42-55_JAX_260_ds1_2gpu_batch1024_s-nerf"
# epoch_number = 4

# project_dir = "JAX_260_ds1_2gpu_batch8192_sat-nerf"
# run_id = "2023-12-03_08-22-34_JAX_260_ds1_2gpu_batch8192_sat-nerf"
# epoch_number = 32


logs_dir = f"{base_dir}/{project_dir}/logs"
checkpoints_dir = f"{base_dir}/{project_dir}/checkpoints"
# eval_aoi(run_id, logs_dir, output_dir, epoch_number, "val", checkpoints_dir, root_dir, img_dir, gt_dir, 0.0)
# eval_aoi(run_id, logs_dir, output_dir, epoch_number, "val", checkpoints_dir, root_dir, img_dir, gt_dir, 0.3)
# eval_aoi(run_id, logs_dir, output_dir, epoch_number, "val", checkpoints_dir, root_dir, img_dir, gt_dir, 0.5)
# eval_aoi(run_id, logs_dir, output_dir, epoch_number, "val", checkpoints_dir, root_dir, img_dir, gt_dir, 0.7)
eval_aoi(run_id, logs_dir, output_dir, epoch_number, "val", checkpoints_dir, root_dir, img_dir, gt_dir, 0.9)
