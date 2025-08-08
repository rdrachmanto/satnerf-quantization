#!/bin/bash


models=(
  # "model_int8"
  "model_group1_int8"
  "model_group2_int8"
  "model_group3_int8"
  "model_group4_int8"
  "model_group5_int8"
  "model_group6_int8"
  "model_group7_int8"
  "model_group8_int8"
  "model_group1_group2_int8"
  "model_group2_group3_int8"
  "model_group3_group4_int8"
  "model_group4_group5_int8"
  "model_group5_group6_int8"
  "model_group6_group7_int8"
  "model_group7_group8_int8"
  "model_group1_group2_group3_int8"
  "model_group2_group3_group4_int8"
  "model_group3_group4_group5_int8"
  "model_group4_group5_group6_int8"
  "model_group5_group6_group7_int8"
  "model_group6_group7_group8_int8"
  "model_group1_group2_group3_group4_int8"
  "model_group2_group3_group4_group5_int8"
  "model_group3_group4_group5_group6_int8"
  "model_group5_group6_group7_group8_int8"
)

for model in "${models[@]}"
do
  python3 -m quantization.profiler --engine "./generated/${model}.trt" > "./logs/layer_info/${model}.json"
done
