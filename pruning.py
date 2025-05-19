import sys
import time
import torch
import os
import json
import metrics
import numpy as np
import sat_utils
import argparse
import glob
import shutil
# from torch.nn.utils import prune
# from torch.nn.utils.prune import random_unstructured, l1_unstructured, global_unstructured, \
#     random_structured, ln_structured
import torch_pruning as tp
from datasets import SatelliteDataset
from eval_satnerf import load_nerf, batched_inference, predefined_val_ts, save_nerf_output_to_images


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

@torch.no_grad()
def count_nonzero_parameters(model):
    for p in model.parameters():
        print(torch.count_nonzero(p).item())
    return sum(torch.count_nonzero(p).item() for p in model.parameters())


def save_model_layers(models, parameters, ratio):
    torch.save(parameters[:1], f"exp-pruning/params-prune-{ratio}-non-sparse-1L.pth")
    torch.save(parameters[:2], f"exp-pruning/params-prune-{ratio}-non-sparse-2L.pth")
    torch.save(parameters[:3], f"exp-pruning/params-prune-{ratio}-non-sparse-3L.pth")
    torch.save(parameters[:4], f"exp-pruning/params-prune-{ratio}-non-sparse-4L.pth")
    torch.save(parameters[:5], f"exp-pruning/params-prune-{ratio}-non-sparse-5L.pth")
    torch.save(parameters[:6], f"exp-pruning/params-prune-{ratio}-non-sparse-6L.pth")
    torch.save(parameters[:7], f"exp-pruning/params-prune-{ratio}-non-sparse-7L.pth")
    torch.save(parameters[:8], f"exp-pruning/params-prune-{ratio}-non-sparse-8L.pth")
    torch.save(parameters[:8], f"exp-pruning/orig-params-16L.pth")
    torch.save(models["coarse"].state_dict(), f"exp-pruning/{run_id}-prune-{ratio}.ckpt")


# def structured_pytorch(model, ratio=0.5):
#     parameters = [model.fc_net[i] for i in range(2, len(model.fc_net)-1, 2)]
#     # Structured pruning
#     start = time.time()
#     for p in parameters:
#         # random_structured(p, "weight", amount=prune_amount, dim=0)
#         ln_structured(p, "weight", amount=ratio, n=2, dim=0)
#         # ln_structured(p, "weight", amount=prune_amount, n=2, dim=0)
#         prune.remove(p, "weight")
#     pruning_latency = time.time() - start
#     return pruning_latency
#
#
# def unstructured_pytorch(model, ratio=0.5):
#     if ratio == 0.0:
#         return 0.0
#     parameters = [model.fc_net[i] for i in range(2, len(model.fc_net), 2)]
#
#     # # Unstructured pruning
#     # start = time.time()
#     # for p in parameters:
#     #     # random_unstructured(p, "weight", 0.5)
#     #     l1_unstructured(p, "weight", 0.5)
#     start = time.time()
#     global_unstructured([(p, "weight") for p in parameters], pruning_method=prune.L1Unstructured, amount=ratio)
#
#     example_inputs = {
#         "input_xyz": torch.randn(size=[1, 3]).cuda(),
#         "input_direction": None,
#         "input_sun_direction": torch.randn(size=[1, 3]).cuda(),
#         "input_t": torch.randn(size=[1, 4]).cuda(),
#         "sigma_only": False
#     }
#
#     # Remove pruning masks
#     for p in parameters:
#         # print(p.parameters())
#         # print(p.weight)
#
#         weight_mask = list(p.named_buffers())[0][1]
#         weight_count = weight_mask.sum(1)
#         idx_to_prune = torch.topk(weight_count, k=2, largest=False).indices.detach().cpu().numpy()
#         prune.remove(p, "weight")
#         DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
#
#         # 2. Group coupled layers for model.conv1
#         group = DG.get_pruning_group(p, tp.prune_linear_out_channels, idxs=idx_to_prune)
#
#         # 3. Prune grouped layers altogether
#         if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
#             group.prune()
#
#         # for pp in p.parameters():
#         #     pp.weight = torch.nn.Parameter(pp.to_sparse())
#             # print(p.weight.values().shape)
#             # print(p.detach().shape)
#             # print(type(p.to_sparse()))
#     pruning_latency = time.time() - start
#     return pruning_latency


def prune_depgraph(model, importance_criterion, arch, target_module="fc_net", ratio=0.5):
    if ratio == 0.0:
        return 0.0

    print("[pruning.prune_depgraph:79] Run depgraph with ratio ", ratio)
    # 1. Importance criterion

    # # Normalizer for group norm
    # # "sum", "standarization", "mean", "max", 'gaussian', 'sentinel', 'lamp'
    # # Group reduction for group norm:
    # # "sum", "mean", "max", "prod", 'first', 'gate'
    # imp = tp.importance.GroupNormImportance(p=1)
    # imp = tp.importance.GroupNormImportance(p=2)
    # imp = tp.importance.GroupTaylorImportance()
    # imp = tp.importance.GroupHessianImportance()
    # imp = tp.importance.RandomImportance()
    # imp = tp.importance.LAMPImportance()

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    if target_module == "all":
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features in [1, 3]:
                ignored_layers.append(m)  # DO NOT prune final layers!
    else:
        ignored_layers.extend([
            model.fc_net[0],
            model.fc_net[-2:],
            model.sigma_from_xyz,
            model.feats_from_xyz,
            model.rgb_from_xyzdir,
        ])
        if hasattr(model, "sun_v_net"):
            ignored_layers.append(model.sun_v_net)
        if hasattr(model, "sky_color"):
            ignored_layers.append(model.sky_color)
        if hasattr(model, "beta_from_xyz"):
            ignored_layers.append(model.beta_from_xyz)

    # for i in ignored_layers:
    #     print(i)

    example_inputs = {
        "input_xyz": torch.randn(size=[1, 3]).cuda(),
        "input_direction": torch.randn(size=[1, 3]).cuda()
    }
    if arch == "s-nerf":
        example_inputs = {
            "input_xyz": torch.randn(size=[1, 3]).cuda(),
            "input_direction": None,
            "input_sun_direction": torch.randn(size=[1, 3]).cuda(),
        }
    elif arch == "sat-nerf":
        example_inputs = {
            "input_xyz": torch.randn(size=[1, 3]).cuda(),
            "input_direction": None,
            "input_sun_direction": torch.randn(size=[1, 3]).cuda(),
            "input_t": torch.randn(size=[1, 4]).cuda(),
            "sigma_only": False
        }

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance_criterion,             # importance criterion
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )

    # 3. Prune & finetune the model
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    # Taylor expansion requires gradients for importance estimation
    if isinstance(importance_criterion, tp.importance.GroupTaylorImportance):
        loss = model(example_inputs).sum()  # A dummy loss, please replace this line with your loss function and data!
        loss.backward()  # before pruner.step()

    start = time.time()
    pruner.step()
    pruning_latency = time.time() - start

    # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # print(base_macs, base_nparams, macs, nparams)
    return pruning_latency


def save_model(model, dir, name):
    model["coarse"].zero_grad()
    torch.save(model, f"{dir}/{name}.pth")


def load_dataset(args, split="val"):
    dataset = SatelliteDataset(args.root_dir,
                               args.img_dir,
                               split=split,
                               img_downscale=args.img_downscale,
                               cache_dir=args.cache_dir)

    if split == "train":
        with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(1, len(dataset))
    return dataset, samples_to_eval


def evaluate(models, args, dataset, samples_to_eval, out_dir, epoch_number, split="val"):
    psnr, ssim, mae = [], [], []
    inference_times = []
    for i in samples_to_eval:
        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        print("Current allocated memory", torch.cuda.memory_allocated())

        # print(rays.shape, rgbs.shape)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        # print("[pruning.eval_aoi:177] rays.shape, rgbs.shape:", rays.shape, rgbs.shape)
        src_id = sample["src_id"]

        # get W and H, either from sample["w"] and sample["h"] or sqrt of ray shape
        if "h" in sample and "w" in sample:
            print(sample)
            W, H = sample["w"], sample["h"]
            print("if true")
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))
            print("if false")

        print(H, W)
        # Only for sat-nerf, with embedding
        ts = None
        if args.model == "sat-nerf":
            if split == "val":
                t = predefined_val_ts(src_id)
                ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
            else:
                ts = sample["ts"].cuda().squeeze()

        print("BATCHED INFERENCE")
        start = time.time()
        results = batched_inference(models, rays, ts, args)
        inference_times.append(time.time() - start)
        # print("Inference time: ", time.time() - start)
        # break

        # TODO: generate images to their own directory
        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        # out_dir = os.path.join(output_dir, run_id, split)
        os.makedirs(out_dir, exist_ok=True)

        print("SAVE OUTPUT TO IMAGES")
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

        torch.cuda.empty_cache()
        del rays, rgbs, ts, results

    return np.round(np.mean(np.array(psnr)), 2), \
        np.round(np.mean(np.array(ssim)), 2), \
        np.round(np.mean(np.array(mae)), 2), \
        np.round(max(inference_times), 2)


scene_arch = [
    # {
    #     "scene": "JAX_416",
    #     "base_dir": "exps",
    #     "arch": "nerf",
    #     "project_dir": f"JAX_416_nerf_16layers",
    #     "run_id": "2024-01-22_01-47-21_JAX_416_nerf_16layers",
    #     "epoch_number": 24
    # },
    {
        "scene": "JAX_416",
        "base_dir": "exps",
        "arch": "s-nerf",
        "project_dir": f"JAX_416_s-nerf_16layers",
        "run_id": "2024-01-22_20-23-05_JAX_416_s-nerf_16layers",
        "epoch_number": 24
    },
    # {
    #     "scene": "JAX_416",
    #     "base_dir": "exps",
    #     "arch": "sat-nerf",
    #     "project_dir": f"JAX_416_sat-nerf_16layers",
    #     "run_id": "2024-01-23_01-53-42_JAX_416_sat-nerf_16layers",
    #     "epoch_number": 24
    # },
    # {
    #     "scene": "JAX_280",
    #     "base_dir": "exps",
    #     "arch": "nerf",
    #     "project_dir": f"JAX_280_nerf_16layers",
    #     "run_id": "2024-01-21_23-24-27_JAX_280_nerf_16layers",
    #     "epoch_number": 24
    # },
    # {
    #     "scene": "JAX_280",
    #     "base_dir": "exps",
    #     "arch": "s-nerf",
    #     "project_dir": f"JAX_280_s-nerf_16layers",
    #     "run_id": "2024-01-22_04-27-24_JAX_280_s-nerf_16layers",
    #     "epoch_number": 24
    # },
    # {
    #     "scene": "JAX_280",
    #     "base_dir": "exps",
    #     "arch": "sat-nerf",
    #     "project_dir": f"JAX_280_sat-nerf_16layers",
    #     "run_id": "2024-01-22_16-57-51_JAX_280_sat-nerf_16layers",
    #     "epoch_number": 24
    # },
    # {
    #     "scene": "JAX_070",
    #     "base_dir": "exps",
    #     "arch": "nerf",
    #     "project_dir": f"JAX_070_nerf_16layers",
    #     "run_id": "2024-01-15_23-07-45_JAX_070_nerf",
    #     "epoch_number": 28
    # },
    # {
    #     "scene": "JAX_070",
    #     "base_dir": "exps",
    #     "arch": "s-nerf",
    #     "project_dir": f"JAX_070_s-nerf_16layers",
    #     "run_id": "2024-01-20_21-40-35_JAX_070_s-nerf",
    #     "epoch_number": 28
    # },
    # {
    #     "scene": "JAX_070",
    #     "base_dir": "exps",
    #     "arch": "sat-nerf",
    #     "project_dir": f"JAX_070_sat-nerf_16layers",
    #     "run_id": "2024-01-21_02-58-00_JAX_070_sat-nerf",
    #     "epoch_number": 28
    # }
]


def get_args(exp_dict):
    scene = exp_dict["scene"]
    base_dir = exp_dict["base_dir"]
    project_dir = exp_dict["project_dir"]
    run_id = exp_dict["run_id"]

    root_dir = f"/data/datasets/Track3-preprocess/{scene}/ba"
    img_dir = f"/data/datasets/Track3-preprocess/{scene}/ba/crops"
    gt_dir = f"/data/datasets/Track3-Truth-JAX"
    cache_dir = f"/data/datasets/Track3-preprocess/{scene}/ba/cache"

    logs_dir = f"{base_dir}/{project_dir}/logs"
    checkpoints_dir = f"{base_dir}/{project_dir}/checkpoints"

    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    # # load pretrained nerf
    # if checkpoints_dir is None:
    #     checkpoints_dir = args.ckpts_dir
    print(args.batch_size)
    return args, checkpoints_dir, logs_dir


if __name__ == "__main__":
    # output_dir = "exp-pruning"
    output_dir = "exp-pruning-2"

    ratios = [
        0.0,
        # 0.1,
        # 0.2,
        # 0.3,
        # 0.4,
        # 0.5,
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9
    ]
    imps = {
        "GroupNorm_1": tp.importance.GroupNormImportance(p=1),
        # "GroupNorm_2": tp.importance.GroupNormImportance(p=2),
        # "Random": tp.importance.RandomImportance(),
        # "LAMP": tp.importance.LAMPImportance()
    }
    target_layers = [
        # "fc_net",
        "all"
    ]

    eval_results = {}
    for target in target_layers:
        for exp in scene_arch:
            args, checkpoints_dir, logs_dir = get_args(exp)
            print("LOADING DATASET")
            dataset, samples_to_eval = load_dataset(args)
            print("LOAD DATASET DONE")
            project_dir = exp["project_dir"]
            run_id = exp["run_id"]
            arch = exp["arch"]
            epoch_number = exp["epoch_number"]

            for name, measure in imps.items():
                result = []
                save_dir = f"{project_dir}_{name}_{target}"
                os.makedirs(f"{output_dir}/image-output/{save_dir}", exist_ok=True)
                os.makedirs(f"{output_dir}/saved-models/{save_dir}", exist_ok=True)
                for ratio in ratios:
                    print(f"Processing {save_dir}_{ratio}")
                    model = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)

                    # # pruning_latency = prune_depgraph(model["coarse"], measure, arch, target=target, ratio=ratio)
                    # pruning_latency = unstructured_pytorch(model["coarse"], ratio)
                    # count_nonzero_parameters(model["coarse"])
                    # print(model["coarse"])
                    # save_model(model, f"{output_dir}/saved-models/{save_dir}", f"ratio-{ratio}")

                    out_dir = f"{output_dir}/image-output/{save_dir}/ratio-{ratio}"
                    mean_psnr, mean_ssim, mean_mae, inference_time = evaluate(model, args, dataset, samples_to_eval,
                                                                              out_dir, epoch_number)
                    print(mean_psnr, mean_ssim, mean_mae, inference_time)
                #     result.append([ratio, mean_psnr, mean_mae, inference_time, np.round(pruning_latency, 2)])
                # # eval_results[f"{project_dir}_{name}_{target}"] = result
                # results_str = []
                # for row in result:
                #     results_str.append(",".join([str(item) for item in row]))
                # with open(f"{output_dir}/metric/{project_dir}_{name}_{target}.csv", "w") as f:
                #     # print(f"Writing to pruning-result/metric/{project_dir}_{name}_{target}.csv")
                #     f.write("ratio,psnr,mae,inference time,pruning time\n")
                #     f.write("\n".join(results_str))



