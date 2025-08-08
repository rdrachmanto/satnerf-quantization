#!/bin/bash
# -----------------------
# SENSITIVITY CHECK LOOP
# -----------------------
scenes=(
  "JAX_004"
  "JAX_070"
  "JAX_072"
  "JAX_260"
  "JAX_412"
  "JAX_467"
  "JAX_474"
)
layer_groups=(
  "group1"
  "group2"
  "group3"
  "group4"
  "group5"
  "group6"
  "group7"
  "group8"
  "group1,group2"
  "group2,group3"
  "group3,group4"
  "group4,group5"
  "group5,group6"
  "group6,group7"
  "group7,group8"
  "group1,group2,group3"
  "group2,group3,group4"
  "group3,group4,group5"
  "group4,group5,group6"
  "group5,group6,group7"
  "group6,group7,group8"
  "group1,group2,group3,group4"
  "group2,group3,group4,group5"
  "group3,group4,group5,group6"
  "group4,group5,group6,group7"
  "group5,group6,group7,group8"
)
precisions=("fp16" "int8")

for scene in "${scenes[@]}"; do
  run_id=$(jq -r ".${scene}.run_id" ptq_data_pairs.json)
  logs_dir=$(jq -r ".${scene}.logs_dir" ptq_data_pairs.json)
  epochs=$(jq -r ".${scene}.epochs" ptq_data_pairs.json)

  # create unique log filenames
  timestamp=$(date +"%y%m%d_%h%m%s")
  logfile="logs/sens/${scene}_${timestamp}.txt"
  errfile="logs/err.txt"
  onnxfile="generated/model.onnx"

  # see baseline and create onnx file
  CUDA_VISIBLE_DEVICES=1 python3 eval_satnerf.py \
    -ri "${run_id}" \
    -ld "${logs_dir}" \
    -e "${epochs}" \
    1>> "$logfile" 2> "$errfile"
  echo "baseline and onnx creation done"

  for (( i=0; i<1; i++ )); do
    # conversion to various precisions and check accuracy
    for prc in "${precisions[@]}"; do
      echo "=========== conversions ===========" 1>> "$logfile" 2>> "$errfile"
      for (( j=0; j<${#layer_groups[@]}; j++ )); do
        if [ -f "./generated/calib.cache" ]; then
          rm "./generated/calib.cache"
        fi
        layer_group_cleaned="${layer_groups[$j]//,/_}"
        CUDA_VISIBLE_DEVICES=1 python3 -m quantization.sensitivity_check \
          -p "$onnxfile" \
          -o "generated/model_${layer_group_cleaned}_${prc}.trt" \
          --precision "${prc}" \
          -lgn "${layer_groups[$j]}" \
          1>> "$logfile" 2>> "$errfile"
      done
    done
    echo "tensorrt engines created"

    # Accuracy check
    for prc in "${precisions[@]}"; do
      echo "=========== accuracy tests ===========" 1>> "$logfile" 2>> "$errfile"
      for (( j=0; j<${#layer_groups[@]}; j++ )); do
        layer_group_cleaned="${layer_groups[$j]//,/_}"
        CUDA_VISIBLE_DEVICES=1 python3 eval_satnerf_trt.py \
          -tp "generated/model_${layer_group_cleaned}_${prc}.trt" \
          -ri "${run_id}" \
          -ld "${logs_dir}" \
          -e "${epochs}" \
          1>> "$logfile" 2>> "$errfile"
      done
    done
    echo "tensorrt engines evaluated"
  done
done
