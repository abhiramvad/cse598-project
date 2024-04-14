import onnxruntime as ort
import numpy as np

session_options = ort.SessionOptions()
session_options.enable_profiling = True

session = ort.InferenceSession("pruned_resnet18.onnx", sess_options=session_options)
onnx_input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
outputs = session.run([output_name], {input_name: onnx_input_tensor})

# Get the profiling file
profile_file = session.end_profiling()
print(f"Profiling output saved to: {profile_file}")
