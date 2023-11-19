'''
Please make sure that your AnimalNetwork class is defined in the network.py file,
 and the get_loader function is defined in the dataset.py file.
Additionally, ensure that the paths and filenames for saving/loading
 models are correct for your project structure.
'''
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader  # Make sure to have a dataset.py file with the get_loader function
from network import AnimalNetwork  # Assuming your network is in a file named network.py
from plot_losses import PlotLosses

# Devuelve el costo promedio por lote en el conjunto de validacion
def validation_step(val_loader, net, cost_function):
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_labels = batch['label']
        device = net.device
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.inference_mode():
            logits, _ = net(batch_imgs)
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    return average_val_loss

def train():
    learning_rate = 1e-4
    n_epochs = 100
    batch_size = 256

    # Transforms for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Assuming your images are 64x64
        transforms.ToTensor(),
    ])

    # Train, validation loaders
    train_dataset, train_loader = get_loader("train", batch_size=batch_size, transform=transform, shuffle=True)
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, transform=transform, shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()

    # Model instantiation
    modelo = AnimalNetwork(input_channels=3,  # Assuming 3 channels for color images
                           n_classes=len(train_dataset.classes))  # Number of classes in your dataset

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

    best_epoch_loss = np.inf

    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed']
            batch_labels = batch['label']
            batch_imgs = batch_imgs.to(modelo.device)
            batch_labels = batch_labels.to(modelo.device)

            optimizer.zero_grad()  # Zero the gradients
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimizer step

            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        # GUARDADO DEL MEJOR MODELO
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            modelo.save_model("best_model.pth")
        # ACTUALIZACION DE GRAFICA
        plotter.on_epoch_end(epoch, train_loss, val_loss)

    plotter.on_train_end()

if __name__ == "__main__":
    train()
