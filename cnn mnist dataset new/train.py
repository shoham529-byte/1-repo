import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNIST_CNN

def train_model(epochs=5, batch_size=64, learning_rate=0.001, save_path="mnist_cnn.pth"):
   
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 
    model.train() 
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

         
            optimizer.zero_grad()

          
            outputs = model(data)
            loss = criterion(outputs, target)

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 100 == 99:    
                print(f"Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx + 1} | Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_model()