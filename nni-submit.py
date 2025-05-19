from nni import Experiment
from nni.experiment import ExperimentConfig, AlgorithmConfig, LocalConfig


# oma_list = [
#     ("OMA_203", 1, 8090), ("OMA_211", 2, 8091),
#     ("OMA_212", 1, 8092), ("OMA_221", 2, 8093),
#     ("OMA_225", 1, 8094), ("OMA_230", 2, 8095),
#     ("OMA_244", 1, 8096), ("OMA_247", 2, 8097),
#     ("OMA_248", 1, 8098), ("OMA_251", 2, 8099),
#     ("OMA_258", 1, 8100), ("OMA_269", 2, 8101),
#     ("OMA_276", 1, 8102), ("OMA_278", 2, 8103),
#     ("OMA_281", 1, 8104), ("OMA_287", 2, 8105),
#     ("OMA_288", 1, 8106), ("OMA_292", 2, 8107),
#     ("OMA_315", 1, 8108), ("OMA_329", 2, 8109),
#     ("OMA_331", 1, 8110), ("OMA_332", 2, 8111),
#     ("OMA_342", 1, 8112), ("OMA_353", 2, 8113),
#     ("OMA_355", 1, 8114), ("OMA_357", 2, 8115),
#     ("OMA_364", 1, 8116), ("OMA_367", 2, 8117),
#     ("OMA_374", 1, 8118), ("OMA_376", 2, 8119),
#     ("OMA_381", 1, 8120), ("OMA_383", 2, 8121),
#     ("OMA_389", 1, 8122), ("OMA_391", 2, 8123)
# ]

oma_list = [
    ("203", 8090), ("211", 8091),
    ("212", 8092), ("221", 8093),
    # ("225", 8094), ("230", 8095),
    # ("244", 8096), ("247", 8097),
    # ("248", 8098), ("251", 8099),
    # ("258", 8100), ("269", 8101),
    # ("276", 8102), ("278", 8103),
    # ("281", 8104), ("287", 8105),
    # ("288", 8106), ("292", 8107),
    # ("315", 8108), ("329", 8109),
    # ("331", 8110), ("332", 8111),
    # ("342", 8112), ("353", 8113),
    # ("355", 8114), ("357", 8115),
    # ("364", 8116), ("367", 8117),
    # ("374", 8118), ("376", 8119),
    # ("381", 8120), ("383", 8121),
    # ("389", 8122), ("391", 8123)
]

script_template = """
#!/usr/bin/env bash

set +e

export LD_LIBRARY_PATH=/home/myid/zis35724/.conda/envs/satnerf/lib:$LD_LIBRARY_PATH

PROJECT_DIR=/data
EXP_DIR=/data/zis35724/satnerf/exp-nni-oma/

MAX_TRAIN_STEPS=2580
BATCH_SIZE=8192
CHUNK=40960
SCENE=OMA_xxx

EXP_NAME="${SCENE}_NNI_$(date +"%Y%m%d_%H%M%S")"

if [ ! -d "$EXP_DIR/$EXP_NAME" ]; then
  mkdir "$EXP_DIR/$EXP_NAME"
fi

python3 -W ignore::LightningDeprecationWarning main-nni.py \\
  --root_dir $PROJECT_DIR/datasets/Track3-preprocess-oma/${SCENE}/ba \\
  --img_dir $PROJECT_DIR/datasets/Track3-preprocess-oma/${SCENE}/ba/crops \\
  --gt_dir $PROJECT_DIR/datasets/Track3-Truth-JAX \\
  --exp_name $EXP_NAME \\
  --model sat-nerf \\
  --img_downscale 1 \\
  --cache_dir $PROJECT_DIR/datasets/Track3-preprocess-oma/${SCENE}/ba/cache \\
  --logs_dir $EXP_DIR/$EXP_NAME/logs \\
  --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \\
  --max_train_steps $MAX_TRAIN_STEPS \\
  --batch_size $BATCH_SIZE \\
  --chunk $CHUNK \\
  --fc_units 256 2>> $EXP_DIR/$EXP_NAME/outputs.txt
"""

for scene_number, port in oma_list:
    gpu_idx = 2

    script_name = f"train-nni-{scene_number}.sh"

    with open(script_name, "w") as f:
        f.write(script_template.replace("SCENE=OMA_xxx", f"SCENE=OMA_{scene_number}"))

    config = ExperimentConfig(
        experiment_name=f"sat_nas_TPE_nni_oma{scene_number}",
        search_space={
            "layers": {"_type": "choice", "_value": [6, 8, 10, 4]},
            "feat": {"_type": "choice", "_value": [128, 256, 64]},
            "lr": {"_type": "choice", "_value": [5e-4, 1e-4, 2e-4]},
            "std": {"_type": "choice", "_value": [0.0, 0.12, 0.14]},
            "n_samples": {"_type": "choice", "_value": [64, 32, 16, 72]}
        },
        tuner=AlgorithmConfig(
            name="TPE",
            class_args={
                "optimize_mode": "maximize"
            }
        ),
        training_service=LocalConfig(
            use_active_gpu=False,
            gpu_indices=[gpu_idx]
        ),
        trial_code_directory=".",
        trial_command=f"bash {script_name}",
        trial_concurrency=1,
        trial_gpu_number=1,
        max_experiment_duration="20h",
        max_trial_number=50,
        use_annotation=False,

    )
    experiment = Experiment(config)
    experiment.run(port=port, wait_completion=True)