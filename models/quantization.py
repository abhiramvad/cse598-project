import torch
import torch.nn as nn
import torch.quantization
from torchvision.models import resnet18
import numpy as np
import copy
import time
from torch.profiler import profile, record_function, ProfilerActivity

# Load and prepare the original model
model_fp32 = resnet18(pretrained=True)
model_fp32.eval()
torch.backends.quantized.engine = 'qnnpack'

def profile_model(model, inputs):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Function to apply dynamic quantization
def dynamic_quantization(model):
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    return model_quantized

# Apply quantization
model_dynamic = dynamic_quantization(model_fp32)
model_dynamic.eval()

torch.save(model_dynamic.state_dict(), "quantized_resnet18.pt")

# Function to measure inference speed
def measure_inference_speed(model, input_tensor, iterations=100):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Create a random input tensor
input_tensor = torch.randn(1, 3, 224, 224)


# Profile dynamic quantized model
profile_model(model_dynamic, input_tensor)


