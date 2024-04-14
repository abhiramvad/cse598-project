import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
import torch.onnx
import onnxruntime as ort
import copy

# Function to apply pruning to all Conv2d layers
def apply_pruning_to_all_conv_layers(model, amount=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', amount=amount)

# Load and prepare the original model
unpruned_model = resnet18(pretrained=True)
unpruned_model.eval()

# Clone and prune the model
model = copy.deepcopy(unpruned_model)
model.eval()
apply_pruning_to_all_conv_layers(model, amount=0.5)  # 50% pruning

# Function to measure inference speed and use profiler
def measure_inference_speed_with_profiling(session, input_tensor, iterations=5):
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {'input': input_tensor})
    end_time = time.time()
    return (end_time - start_time) / iterations

# Prepare input tensor
input_tensor = torch.randn(1, 3, 224, 224).detach().numpy()

# Export both models to ONNX
torch.onnx.export(unpruned_model, torch.from_numpy(input_tensor), "unpruned_resnet18.onnx", opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
torch.onnx.export(model, torch.from_numpy(input_tensor), "pruned_resnet18.onnx", opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Enable profiling and measure speed
unpruned_options = ort.SessionOptions()
unpruned_options.enable_profiling = True
unpruned_options.profile_file_prefix = "unpruned_model_profile"

pruned_options = ort.SessionOptions()
pruned_options.enable_profiling = True
pruned_options.profile_file_prefix = "pruned_model_profile"

unpruned_session = ort.InferenceSession("unpruned_resnet18.onnx", unpruned_options)
pruned_session = ort.InferenceSession("pruned_resnet18.onnx", pruned_options)

unpruned_speed = measure_inference_speed_with_profiling(unpruned_session, input_tensor)
pruned_speed = measure_inference_speed_with_profiling(pruned_session, input_tensor)

# Retrieve and print profiling information
unpruned_profile = unpruned_session.end_profiling()
pruned_profile = pruned_session.end_profiling()

print(f"Inference time with ONNX Runtime (Unpruned): {unpruned_speed:.5f} seconds")
print(f"Inference time with ONNX Runtime (Pruned): {pruned_speed:.5f} seconds")
print(f"Profiler data (Unpruned): {unpruned_profile}")
print(f"Profiler data (Pruned): {pruned_profile}")
