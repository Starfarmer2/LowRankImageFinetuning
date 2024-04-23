import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Example layer
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*32*32, 10)  # Assuming input images are 32x32

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Initialize the model
model = SimpleNN()

# Create an input tensor, with requires_grad=True to allow gradient computation
input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)  # Batch size of 1, 3 color channels, 32x32

# Set up an optimizer that will update the input tensor
optimizer = optim.Adam([input_tensor], lr=0.01)  # Learning rate is quite arbitrary here

# Define a target class, e.g., class index 2
target_class = 2

# Use CrossEntropyLoss for a classification task
loss_function = nn.CrossEntropyLoss()


for i in range(100):  # Number of optimization steps
    optimizer.zero_grad()  # Clear previous gradients
    output = model(input_tensor)  # Forward pass: compute predictions

    # Create a target tensor to calculate loss
    target = torch.LongTensor([target_class])  # Target is the class we want to maximize

    loss = loss_function(output, target)  # Compute loss to maximize the class score
    loss.backward()  # Compute gradients for input_tensor
    optimizer.step()  # Update input_tensor based on gradients

    if i % 10 == 0:
        print(f'Step {i}, Loss: {loss.item()}')

print(input_tensor)
# After optimization, input_tensor is optimized to maximize the target class prediction
