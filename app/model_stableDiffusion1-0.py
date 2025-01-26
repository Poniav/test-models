"""
Stable Diffusion XL BASE 1.0 : https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
Stable Diffusion XL REFINE 1.0 : https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

// Installer les bibliothèques nécessaires
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adapté à CUDA 11.8
pip install diffusers[torch] transformers accelerate xformers


"""
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
import torch
from PIL import Image

# Load the base SDXL model
base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

# Load the refiner SDXL model
refiner_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
).to("cuda")

# Parameters for the generation
prompt = "A futuristic cityscape at sunset, ultra-realistic, vibrant colors"
negative_prompt = "low quality, blurry, unrealistic"
guidance_scale = 7.5  # Poids pour la guidance
num_inference_steps = 50  # Nombre d'étapes d'inférence

# Generate the base image
base_image = base_pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                       num_inference_steps=num_inference_steps).images[0]

# Save the base image
base_image.save("output_base_image.png")

# Use the refiner to refine the base image
refined_image = refiner_pipe(image=base_image, prompt=prompt, negative_prompt=negative_prompt,
                             guidance_scale=guidance_scale).images[0]

# Save the refined image
refined_image.save("output_refined_image.png")

print("Images générées et enregistrées : 'output_base_image.png' et 'output_refined_image.png'")
