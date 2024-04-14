import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Define the teacher model
teacher_model = models.resnet18(pretrained=True)
teacher_model.eval()

# Define the student model (smaller version of ResNet)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# Define the loss function for knowledge distillation
def knowledge_distillation_loss(outputs, teacher_outputs, labels, alpha=0.1, temperature=5):
    soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
    log_probs = F.log_softmax(outputs / temperature, dim=1)
    kd_loss = -(soft_targets * log_probs).sum(dim=1).mean()
    ce_loss = F.cross_entropy(outputs, labels)
    total_loss = alpha * kd_loss + (1. - alpha) * ce_loss
    return total_loss

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, position=0, leave=True)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        teacher_outputs = teacher_model(inputs).detach()
        loss = criterion(outputs, teacher_outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        pbar.set_description(f"Training Loss: {running_loss / len(train_dataset):.4f}")

# Initialize the student model
student_model = StudentModel()

# Train the student model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)

optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    train(student_model, train_loader, optimizer, criterion, device)
    scheduler.step()

# Save the trained student model
torch.save(student_model.state_dict(), "student_resnet18.pth")

# Convert the student model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
student_model.eval()
torch.onnx.export(student_model, dummy_input, "student_resnet18.onnx", opset_version=11)

print("Student model saved as 'student_resnet18.pth' and 'student_resnet18.onnx'")
