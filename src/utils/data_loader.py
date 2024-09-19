import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from src.utils.config import get_config

def load_mnist(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_set = datasets.MNIST(root=config.data_root, download=True, train=True,
                                transform=transform)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    test_set = datasets.MNIST(root=config.data_root, download=True, train=False, 
                                transform=transform)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader

def main(config):
    train_loader, test_loader = load_mnist(config)
    images, labels = next(iter(train_loader))

    output_dir = 'results/sample_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_images = 5
    for i in range(num_images):
        image = images[i].squeeze()
        label = labels[i]
        plt.figure()
        plt.imshow(image,cmap='gray')
        plt.colorbar()
        plt.title(f'Label: {label}')
        plt.savefig(os.path.join(output_dir, f'mnist_sample_{i}.png'))
        plt.close()



if __name__=='__main__':
    config = get_config()
    main(config)



