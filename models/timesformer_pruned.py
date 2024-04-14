import av
import torch
import numpy as np
import time
import psutil
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune
# from neural_compressor.pruner.pruning import Pruning, WeightPruningConfig
# from neural_compressor import Pruning, WeightPruningConfig

# Initialize TensorBoard writer
writer = SummaryWriter('runs/video_inference')

# Seed for reproducibility
np.random.seed(0)

def checkModules(model):# Checking the modules
    print(model)
    # Assuming `model` is your PyTorch model
    for name, module in model.named_modules():
        print(name, module)

def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
                print("Prunning!!!!!")
            except:
                pass
            try:
                prune.remove(module, "bias")
                print("Prunning!!!!!")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
                print("Prunning!!!!!")
            except:
                pass
            try:
                prune.remove(module, "bias")
                print("Prunning!!!!!")
            except:
                pass
        # elif isinstance(module, torch.nn.MultiheadAttention):
        elif isinstance(module, ):
            try:
                # Prune the attention weights and biases
                prune.remove(module, "in_proj_weight")
                prune.remove(module, "in_proj_bias")
                prune.remove(module, "out_proj.weight")
                prune.remove(module, "out_proj.bias")
                print("Pruning attention weights and biases of", module_name)
            except:
                pass
    return model


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: Decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (int): Total number of frames to sample.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Maximum allowed index of sample's last frame.
    Returns:
        List[int]: List of sampled frame indices.
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices.tolist()

file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
container = av.open(file_path)
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)

image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-hr-finetuned-k400")
# model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-hr-finetuned-k600")

checkModules(model)


# Prunning Query Key Value + Prunning MLP
for i in range(11):
    print("pruning")
    # module = model.timesformer.encoder.layer[i].attention.qkv
    module = model.timesformer.encoder.layer[i].attention.attention.qkv
    prune.random_unstructured(module, name="weight", amount=0.9)
    prune.remove(module, 'weight')  # Make the pruning permanent
    print(list(module.named_parameters()))

    print()
    print("layer norm")
    prune.ln_structured(module, name='weight', amount=0.9, n=2, dim=1)
    prune.remove(module, 'weight')  # Make the pruning permanent
    print("----------")
    print(list(module.named_buffers()))


for i in range(11):
    print("pruning temporal attention")
    module = model.timesformer.encoder.layer[i].temporal_attention.attention.qkv
    # module = model.timesformer.encoder.layer[i].attention.attention.qkv
    prune.random_unstructured(module, name="weight", amount=0.9)
    prune.remove(module, 'weight')  # Make the pruning permanent
    print(list(module.named_parameters()))


for i in range(11):
    print("pruning dense attention")
    module = model.timesformer.encoder.layer[i].output.dense
    # module = model.timesformer.encoder.layer[i].attention.attention.qkv
    prune.random_unstructured(module, name="weight", amount=0.9)
    prune.remove(module, 'weight')  # Make the pruning permanent

    print(list(module.named_parameters()))
    
# timesformer.encoder.layer.11.output.dense Linear(in_features=3072, out_features=768, bias=True)
    
# PATH = "/mnt/storage/ji/TimeSformer/model_size"
# bin = torch.save(model.state_dict(), PATH)

# torch.save(model.state_dict(), 'model_weights.bin')

# Number of parameters not changed
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("params: ", params)

total_params = sum(p.numel() for p in model.parameters())
total_params_size = total_params * 4  # As each parameter is a 32-bit float
print(total_params, total_params_size / (1024**2))  # size in megabytes

inputs = image_processor(list(video), return_tensors="pt")

# Memory and Time Measurement Start
pre_inference_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Memory and Time Measurement End
inference_time = time.time() - start_time
post_inference_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
memory_diff = post_inference_memory - pre_inference_memory

predicted_label = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_label]

# Log metrics to TensorBoard
writer.add_scalar('Inference Time', inference_time, global_step=1)
writer.add_scalar('Memory Usage Before Inference', pre_inference_memory, global_step=1)
writer.add_scalar('Memory Usage After Inference', post_inference_memory, global_step=1)
writer.add_scalar('Memory Usage Difference', memory_diff, global_step=1)

writer.close()

print(f"Predicted class: {predicted_class}")
print(f"Inference time: {inference_time:.2f} seconds")
print(f"Memory usage before inference: {pre_inference_memory:.2f} MB")
print(f"Memory usage after inference: {post_inference_memory:.2f} MB")
print(f"Memory usage difference: {memory_diff:.2f} MB")
