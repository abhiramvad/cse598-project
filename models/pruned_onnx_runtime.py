import onnxruntime as ort
import numpy as np
import time

# Load the ONNX model
session = ort.InferenceSession("pruned_resnet18.onnx")

# Prepare input according to ONNX Runtime requirements
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Function to measure inference speed
def measure_inference_speed_onnx(session, input_tensor, iterations=100):
    start_time = time.time()
    for _ in range(iterations):
        session.run([output_name], {input_name: input_tensor})
    end_time = time.time()
    return (end_time - start_time) / iterations

# Measure the inference speed
onnx_speed = measure_inference_speed_onnx(session, input_tensor)

print(f"Inference time with ONNX Runtime: {onnx_speed:.5f} seconds")
