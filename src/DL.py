import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim



class ConvNet(nn.Module):
    def __init__(self, num_features=1,input_dim=(3,32,32),learning_rate=0.001,kernel=[3,3,3],features_size=[16,32,64],stride=[1,1,1]):
        super(ConvNet, self).__init__()
        assert len(kernel)==len(features_size)==len(stride), "The length of kernel, features_size and stride should be the same"
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim[0]
        
        for k, f, s in zip(kernel, features_size, stride):
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=f,
                kernel_size=k,
                stride=s,
                padding=k//2  # Padding automatique pour maintenir la dimension
            )
            self.conv_layers.append(conv_layer)
            in_channels = f  # Pour la prochaine couche
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatened_size = self._flatened_size(input_dim)
        self.fc1 = nn.Linear(self.flatened_size, 512)
        self.fc2 = nn.Linear(512, num_features)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion=nn.MSELoss()


    def _flatened_size(self, input_dim):
        x=torch.ones(1,input_dim[0],input_dim[1],input_dim[2])
        x = self.pool(F.relu(self.layer1(x)))
        x = self.pool(F.relu(self.layer2(x)))
        x = self.pool(F.relu(self.layer3(x)))
        return torch.prod(torch.tensor(x.shape[1:])).item()

        
    def forward(self, x):
        x = self.pool(F.relu(self.layer1(x)))
        x = self.pool(F.relu(self.layer2(x)))
        x = self.pool(F.relu(self.layer3(x)))
        x = x.view(-1, self.flatened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        output = self(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    # Print system information
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {torch.device('cpu')}\n")
    
    # Create model
    model = ConvNet(num_features=10)
    print("Model Architecture:")
    print(model)
    
    # Test with dummy data
    try:
        x = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
        output = model(x)
        print("\nForward pass successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
