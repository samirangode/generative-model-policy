import torch
import os
import matplotlib.pyplot as plt
from src.utils.config import get_config    
from src.utils.data_loader import load_mnist




if __name__=="__main__":
    config = get_config()
    load_mnist(config=config)