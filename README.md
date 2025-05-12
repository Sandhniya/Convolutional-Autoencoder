# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

STEP 1:
Import necessary libraries including PyTorch, torchvision, and matplotlib.

STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.

STEP 3:
Add Gaussian noise to training and testing images using a custom function.

STEP 4:
Define the architecture of a convolutional autoencoder:

Encoder: Conv2D layers with ReLU + MaxPool

Decoder: ConvTranspose2D layers with ReLU/Sigmoid

STEP 5:
Initialize model, define loss function (MSE) and optimizer (Adam).

STEP 6:
Train the model using noisy images as input and original images as target.

STEP 7:
Visualize and compare original, noisy, and denoised images.

PROGRAM

### Name:SANDHIYA SREE B
### Register Number:212223220093

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # 28x28 -> 14x14
            nn.Conv2d(16, 8, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              # 14x14 -> 7x7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),    # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),    # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),             # 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
```
# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

```
```

def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:SANDHIYA SREE B")
    print("Register Number: 212223220093")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
     
```
        

## OUTPUT

### Model Summary

![image](https://github.com/user-attachments/assets/05b855fa-6b5f-4aae-b46c-915ce9bfb3c4)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/c52406d9-5382-4859-a651-caff82fa6017)




## RESULT

The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
