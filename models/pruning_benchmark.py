import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import time
from torchvision.models import resnet18

# Load a pre-trained model
model = resnet18(pretrained=True)
model.eval()

# Function to apply pruning to a given module
def apply_pruning_to_model(model, amount=0.2):
    # Pruning the first Conv2d layer as an example
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            break  # Prune only the first Conv2d layer found

# Apply pruning
apply_pruning_to_model(model)

# Function to measure the inference speed
def measure_inference_speed(model, inputs, iterations=100):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Create a dummy input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Measure the inference speed before pruning
original_speed = measure_inference_speed(model, input_tensor)

# Check if the pruning is applied and then remove it
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
            break  # Remove pruning from the first pruned layer

# Measure the inference speed after removing pruning
pruned_speed = measure_inference_speed(model, input_tensor)

print(f"Inference time before pruning: {original_speed:.5f} seconds")
print(f"Inference time after removing pruning: {pruned_speed:.5f} seconds")
