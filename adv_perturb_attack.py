import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.mlpmodel import MLP

# Load the trained model
model = MLP()
model.load_state_dict(torch.load('model_weights.pth'))  # Ensure this path is correct
model.eval()

# Prepare MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # Batch size set to 1

# FGSM attack function
def fgsm_attack(image, label, model, lr, num_epochs, L=0.0001):
    image = torch.nn.Parameter(image.clone())
    image.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([image], lr=lr)

    # # Iteratively update weights
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = model(image)
    #     loss = criterion(output, label)
    #     loss.backward()
    #     optimizer.step()

    # Iteratively update weights, end condition loss <= L
    loss = 1
    while loss > L:
        optimizer.zero_grad()
        output = model.forward_sigmoidpreprocess(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f'loss: {loss}')

    # perturbed_image = torch.clamp(image, 0, 1)
    perturbed_image = torch.sigmoid(image)
        
    # output = model(image)
    # loss = F.nll_loss(F.log_softmax(output, dim=1), attack_label)
    # model.zero_grad()
    # loss.backward()
    # data_grad = image.grad.data

    # sign_data_grad = data_grad.sign()
    # perturbed_image = image + epsilon * sign_data_grad
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Process a single image from the test set
data_iter = iter(test_loader)
image, label = data_iter.next()
print(image[0].shape)

# Start with gaussian noise
image = torch.randn(1,1,28,28)

# Custom label
attack_label = torch.tensor([0])

# Generate the adversarial image
lr = 2  # Perturbation magnitude
num_epochs = 5000
perturbed_image = fgsm_attack(image, attack_label, model, lr, num_epochs)

# Display the original and adversarial images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.detach().numpy().squeeze(), cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(perturbed_image.detach().numpy().squeeze(), cmap='gray')
ax[1].set_title("Adversarial Image")
plt.show()

# Test the model's prediction on the adversarial image
output_perturbed = model(perturbed_image)
predicted_label = output_perturbed.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability

print(f"Original Label: {label.item()}")
print(f"Predicted Label: {predicted_label.item()}")
print(f"Output_perturbed: {output_perturbed}")


# Ensemble multiple networks
# Low rank input