import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_load import YourVideoDataset  # Adjusted to provide file paths
from timesformer_v2 import Timesformer
from resnet3d import Resnet3D


# Load models
timesformer_model = Timesformer()
resnet_3d_model = Resnet3D()

# Set modes
timesformer_model.model.eval()  # Teacher in eval mode
resnet_3d_model.model.train()  # Student in train mode

# DataLoader
dataset = YourVideoDataset()  # Ensure this is providing file paths and numeric labels
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch_size as necessary

# Loss functions and optimizer
classification_loss_fn = nn.CrossEntropyLoss()
distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')  # Use batchmean for averaging
optimizer = torch.optim.Adam(resnet_3d_model.model.parameters(), lr=1e-4)

# Hyperparameters
temperature = 5.0
alpha = 0.5  # Weight for distillation loss; adjust as needed

# DataLoader with batch_size=1
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Adjusted Training loop for batch_size=1
for epoch in range(50):
    for video_path, label in dataloader:
        print(f"Labels batch size: {len(label)}")  # Debugging line

        video_path = video_path[0]  # Unpack the batch
        # label = label.to(resnet_3d_model.device)  # Move label to correct device
        # label = label.squeeze()  # Remove unnecessary batch dimension
        
        # No need to loop through video_paths since batch_size=1
        teacher_logits = timesformer_model.predict_logits(video_path)
        student_logits = resnet_3d_model.predict(video_path)
        print(student_logits.requires_grad)  # Should be True
        print(video_path,label)
        # Ensure logits are correctly sized
        # teacher_logits = teacher_logits.unsqueeze(0)  # Add batch dim if not present
        # student_logits = student_logits.unsqueeze(0)  # Add batch dim if not present
        # print(f"Labels shape: {label.shape}")
        # print(f"Student logits shape: {student_logits.shape}")
        # Compute losses
        classification_loss = classification_loss_fn(student_logits, label)  # Label needs batch dim
        distillation_loss = distillation_loss_fn(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        )
        total_loss = alpha * classification_loss + (1 - alpha) * distillation_loss
        print(total_loss.grad_fn)  # Should not be None

        # Backprop and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {total_loss.item()}")

# Save the fine-tuned student model
torch.save(resnet_3d_model.model.state_dict(), 'resnet3d_distilled.bin')
