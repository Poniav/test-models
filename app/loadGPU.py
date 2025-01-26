"""
This script is used to check if a GPU is available and to load the Stable Diffusion pipeline.
The pipeline is loaded from the "runwayml/stable-diffusion-v1-5" model. (https://huggingface.co/runwayml/stable-diffusion-v1-5)
"""

import torch
from diffusers import StableDiffusionPipeline

# Verify the presence of a compatible GPU
if torch.cuda.is_available():
    print("CUDA available: GPU detected")
else:
    print("CUDA unavailable: running on CPU")

# Function to load the pipeline
def loadGPU():
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Pipeline successfully loaded!")

    return pipeline

# Load the pipeline - Uncomment the line below to test
# pipeline = loadGPU()



