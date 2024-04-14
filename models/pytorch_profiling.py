from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models import resnet18
import torch
def profile_model(model, inputs):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Profile the PyTorch model
model = resnet18(pretrained=True)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)
profile_model(model, input_tensor)
