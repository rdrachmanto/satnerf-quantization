import random

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
onnx_path = "/data/rdr78068/satnerf-base/generated/model.onnx"

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# === Parse ONNX ===
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_path, "rb") as model_file:
    if not parser.parse(model_file.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed.")

# === Inspect layers (optional per-layer precision control) ===
available_precisions = (
    trt.DataType.HALF, 
    # trt.DataType.FP8, 
    trt.DataType.BF16
)

for i in range(network.num_layers):
    layer = network.get_layer(i)
    # print(f"Layer {i}: {layer.name}, type: {layer.type}, precision: {layer.precision}")

    gemms = {
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.SCALE,
        trt.LayerType.ELEMENTWISE
    }

    if layer.type in gemms:
        chosen_precision = random.choice(available_precisions)
        layer.precision = chosen_precision
        for j in range(layer.num_outputs):
            layer.set_output_type(j, chosen_precision)

# === Enable Precision Modes ===
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.BF16)
config.set_flag(trt.BuilderFlag.FP8)

# === Define Optimization Profile ===
profile = builder.create_optimization_profile()

# Based on your earlier shape setup
input_shapes = {
    "input_xyz":   ([1, 3], [358656, 3], [358656, 3]),
    "input_sun_dir": ([1, 3], [358656, 3], [358656, 3]),
    "input_t":     ([1, 4], [358656, 4], [358656, 4]),
}

for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
    profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

config.add_optimization_profile(profile)

# === Build the engine ===
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Engine build failed")

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)

# === Save engine to file ===
with open("/data/rdr78068/satnerf-base/generated/model_fp8_fp16_bf16.trt", "wb") as f:
    f.write(engine.serialize())

print("TensorRT engine built successfully!")
