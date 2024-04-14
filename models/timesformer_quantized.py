import av
import torch
import numpy as np
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from huggingface_hub import hf_hub_download
import torch.nn.utils.prune as prune
import torch.onnx
import onnx
from onnx import numpy_helper
from torch.quantization import quantize_dynamic

# Set the backend to FBGEMM or QNNPACK depending on your system architecture
torch.backends.quantized.engine = 'qnnpack'  # Use 'qnnpack' for ARM

class TimesformerQuantized:
    def __init__(self, model_name="facebook/timesformer-hr-finetuned-k400"):
        self.model_name = model_name
        self.model = TimesformerForVideoClassification.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k400")
        # self.apply_pruning()
        # Optionally perform quantization after pruning
        self.quantize_model()

    def quantize_model(self):
        # Perform dynamic quantization on the model
        # Quantizing only the Linear layers since they benefit the most typically from quantization
        self.model = quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def apply_pruning(self):
        # Applying pruning to different components of the encoder layers
        for name, module in self.model.named_modules():
            if 'qkv' in name and isinstance(module, torch.nn.Linear):
                # Apply lighter pruning to QKV layers
                prune.l1_unstructured(module, name='weight', amount=0.9)  # Example: 30% pruning
                prune.remove(module, 'weight')  # Comment this out to keep pruning reparametrization

            elif 'dense' in name and isinstance(module, torch.nn.Linear):
                # Apply more aggressive pruning to dense layers in intermediate and output blocks
                prune.l1_unstructured(module, name='weight', amount=0.9)  # Example: 50% pruning
                prune.remove(module, 'weight')  # Comment this out to keep pruning reparametrization

            # Optionally make pruning permanent (remove mask parameters)
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.9)  # Example: 30% pruning
                prune.remove(module, 'weight')  # Comment this out to keep pruning reparametrization


    def read_video_pyav(self, container, indices):
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

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
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

    def predict_logits(self, file_path):
        container = av.open(file_path)
        indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = self.read_video_pyav(container, indices)

        inputs = self.image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        return logits

    def predict_label(self,file_path):
        logits = self.predict_logits(file_path)
        predicted_label = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_label]
        return predicted_class

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_size = total_params * 4  # As each parameter is a 32-bit float
        return total_params, total_params_size / (1024**2)  # size in megabytes
    
    def get_model_size_and_sparsity(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum(p.nonzero().size(0) for p in self.model.parameters())
        total_params_size = total_params * 4 / (1024 ** 2)  # Convert to megabytes assuming 32-bit floats
        sparsity = 100 * (1 - nonzero_params / total_params)  # Percentage of weights that are zero
        return total_params, total_params_size, sparsity
    
    def calculate_theoretical_memory_savings(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum(p.nonzero().size(0) for p in self.model.parameters())
        total_memory_mb = total_params * 4 / (1024 ** 2)  # Assuming 32-bit floats
        nonzero_memory_mb = nonzero_params * 4 / (1024 ** 2)
        sparsity = 100 * (1 - nonzero_params / total_params)
        return {
            "Total Memory (MB)": total_memory_mb,
            "Nonzero Memory (MB)": nonzero_memory_mb,
            "Sparsity (%)": sparsity
        }






# # Function to convert dense tensor to sparse
# def convert_to_sparse(tensor):
#     indices = torch.nonzero(tensor).t()
#     values = tensor[tensor != 0]
#     shape = tensor.size()
#     return torch.sparse_coo_tensor(indices, values, shape)

# # Export the model to ONNX format
# def export_model(model, input_size, onnx_file_path):
#     model.eval()
#     dummy_input = torch.randn(input_size)
#     torch.onnx.export(model, dummy_input, onnx_file_path)

# # Load the ONNX model and convert to sparse
# def convert_onnx_to_sparse(onnx_file_path):
#     model = onnx.load(onnx_file_path)
#     for tensor in model.graph.initializer:
#         dense_tensor = numpy_helper.to_array(tensor)
#         sparse_tensor = convert_to_sparse(torch.from_numpy(dense_tensor))
#         # Here you would replace the dense tensor with sparse representation
#         # This part is conceptual and may require actual implementation in ONNX format
#     onnx.save(model, onnx_file_path)



# Example usage:
if __name__ == "__main__":
    # file_path = hf_hub_download(
    #     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    # )
    video_classifier = TimesformerQuantized()

    file_path = '../data/test2.mp4'
    print(video_classifier.predict_label(file_path))
    param_count, model_size_mb, sparsity = video_classifier.get_model_size_and_sparsity()
    print(f"Model parameters: {param_count}, Model size: {model_size_mb:.2f} MB, Sparsity: {sparsity:.2f}%")
    # Assuming model is your PyTorch model
    # export_model(video_classifier.model, (1, 3, 224, 224), 'model.onnx')
    # convert_onnx_to_sparse('model.onnx')
    # Usage with your model
    memory_stats = video_classifier.calculate_theoretical_memory_savings()
    print(memory_stats)
