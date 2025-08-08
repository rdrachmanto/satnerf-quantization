#!/bin/bash
# -----------------------
# FULL PTQ LOOP
# -----------------------
scenes=(
  "JAX_004"
  # "JAX_070"
  # "JAX_072"
  # "JAX_260"
  # "JAX_412"
  # "JAX_467"
  # "JAX_474"
)
precisions=("fp16" "int8")

for scene in "${scenes[@]}"; do
  if [ ! -d "./logs/singles" ]; then
    echo "./logs/singles does not exist, creating the folder..."
    mkdir -p ./logs/singles
  fi

  run_id=$(jq -r ".${scene}.run_id" ptq_data_pairs.json)
  logs_dir=$(jq -r ".${scene}.logs_dir" ptq_data_pairs.json)
  epochs=$(jq -r ".${scene}.epochs" ptq_data_pairs.json)

  # create unique log filenames
  timestamp=$(date +"%y%m%d_%h%m%s")
  logfile="logs/singles/${scene}_${timestamp}.txt"
  errfile="logs/err.txt"
  onnxfile="generated/model.onnx"

  # see baseline and create onnx file
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 eval_satnerf.py \
    -ri "${run_id}" \
    -ld "${logs_dir}" \
    -e "${epochs}" \
    # 1>> "$logfile" 2> "$errfile"
  echo "baseline and onnx creation done"

  # for (( i=0; i<1; i++ )); do
  #   # conversion to various precisions and check accuracy
  #   for prc in "${precisions[@]}"; do
  #     echo "=========== conversions ===========" 1>> "$logfile" 2>> "$errfile"
  #     if [ -f "./generated/calib.cache" ]; then
  #       rm "./generated/calib.cache"
  #     fi
  #
  #     python3 -m quantization.full_quant \
  #       -p "$onnxfile" \
  #       -o "generated/model_${prc}.trt" \
  #       --precision "${prc}" \
  #       1>> "$logfile" 2>> "$errfile"
  #   done
  #   echo "tensorrt engines created"
  #
  #   # Accuracy check
  #   for prc in "${precisions[@]}"; do
  #     echo "=========== accuracy tests ===========" 1>> "$logfile" 2>> "$errfile"
  #     CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 eval_satnerf_trt.py \
  #       -tp "generated/model_${prc}.trt" \
  #       -ri "${run_id}" \
  #       -ld "${logs_dir}" \
  #       -e "${epochs}" \
  #       1>> "$logfile" 2>> "$errfile"
  #   done
  #   echo "tensorrt engines evaluated"
  # done
done
