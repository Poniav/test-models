"""
utilsML.py contains utility functions for the Machine Learning model scripts.
1. checkGPU() is a function that checks if a GPU is available.
"""
import torch


# Function to verify the presence of available GPU

def checkGPU():
    if torch.cuda.is_available():
        print("CUDA available: GPU detected")
    else:
        print("CUDA unavailable: running on CPU")

    
