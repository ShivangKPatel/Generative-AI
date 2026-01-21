import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================================
# 1. CONVOLUTION AND POOLING FUNDAMENTALS
# ============================================================================

class ConvolutionDemo(nn.Module):
    """Demonstrates basic convolution and pooling operations"""
    
    def __init__(self):
        super(ConvolutionDemo, self).__init__()
        # Conv layer: input channels=3, output channels=16, kernel size=3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Max pooling: 2x2 kernel, stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Average pooling for comparison
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """Forward pass showing convolution and pooling"""
        conv_out = self.relu(self.conv1(x))
        maxpool_out = self.maxpool(conv_out)
        avgpool_out = self.avgpool(conv_out)
        return conv_out, maxpool_out, avgpool_out


# ============================================================================
# 2. SIMPLE CNN ARCHITECTURE
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Conv -> ReLU -> MaxPool
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 2: Conv -> ReLU -> MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ============================================================================
# 3. TRANSFER LEARNING WITH PRETRAINED MODEL
# ============================================================================

class TransferLearningModel(nn.Module):
    """Transfer learning using pretrained ResNet18"""
    
    def __init__(self, num_classes=10, fine_tune=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze backbone weights if not fine-tuning
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


# ============================================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    
    print(f"Using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    simple_cnn = SimpleCNN(num_classes=10).to(device)
    transfer_model = TransferLearningModel(num_classes=10, fine_tune=True).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    # Train Simple CNN
    print("\n" + "="*50)
    print("Training Simple CNN")
    print("="*50)
    optimizer = optim.Adam(simple_cnn.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        loss = train_epoch(simple_cnn, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Train Transfer Learning Model
    print("\n" + "="*50)
    print("Training Transfer Learning Model (ResNet18)")
    print("="*50)
    optimizer = optim.Adam(transfer_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        loss = train_epoch(transfer_model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Evaluation
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    simple_cnn_metrics = evaluate(simple_cnn, test_loader, device)
    transfer_metrics = evaluate(transfer_model, test_loader, device)
    
    print("\nSimple CNN Performance:")
    for metric, value in simple_cnn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTransfer Learning (ResNet18) Performance:")
    for metric, value in transfer_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()