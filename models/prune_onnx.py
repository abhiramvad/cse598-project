import torch.onnx
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Load and prune the model
model = resnet18(pretrained=True)
model.eval()

# Pruning the first Conv2d layer
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.5)
        break

# Dummy input as required by the export function
input_tensor = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(model, input_tensor, "pruned_resnet18.onnx", opset_version=12,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
