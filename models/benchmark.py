import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
import torch.onnx
import onnxruntime as ort

def apply_pruning_to_all_conv_layers(model, amount=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', amount=amount)

# Load and prepare the original model
model = resnet18(pretrained=True)
model.eval()

# Apply pruning to all Conv2d layers
apply_pruning_to_all_conv_layers(model, amount=0.5)  # Increase pruning to 50%

# Export pruned model to ONNX
input_tensor = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input_tensor, "pruned_resnet18.onnx", opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Measure inference speed for PyTorch
def measure_inference_speed_pytorch(model, inputs, iterations=100):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Measure inference speed for ONNX Runtime
def measure_inference_speed_onnx(session, input_tensor, iterations=100):
    start_time = time.time()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(iterations):
        session.run([output_name], {input_name: input_tensor})
    end_time = time.time()
    return (end_time - start_time) / iterations

# Benchmarking
torch_speed = measure_inference_speed_pytorch(model, input_tensor)
session = ort.InferenceSession("pruned_resnet18.onnx")
onnx_input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
onnx_speed = measure_inference_speed_onnx(session, onnx_input_tensor)

# Results
print(f"Inference time with PyTorch: {torch_speed:.5f} seconds")
print(f"Inference time with ONNX Runtime: {onnx_speed:.5f} seconds")



