"""
Stable Diffusion XL BASE 1.0 : https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
Stable Diffusion XL REFINE 1.0 : https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

// Installer les bibliothèques nécessaires
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adapté à CUDA 11.8
pip install diffusers[torch] transformers accelerate xformers


"""
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image

# Path output: output/output_base_SDXL1_0.png
path = "output/"

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model (Text-to-Image)
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to(device)

# Load the refiner model (Image-to-Image)
refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
).to(device)

# Parameters for generation
#prompt = "A futuristic cityscape at sunset, ultra-realistic, vibrant colors"
prompt = "A hyper-realistic portrait of a beautiful ginger girl with freckles, detailed skin texture, expressive green eyes, soft natural lighting, cinematic sunset in the background, 8K resolution, ultra-detailed, photorealistic, focus on face, dynamic composition, rim lighting, mystical atmosphere"
negative_prompt = "low quality, blurry, unrealistic, cartoonish, painting, drawing, 2D, abstract, deformed, oversaturated"
guidance_scale = 8.5
num_inference_steps = 80

# Generate the base image (Text-to-Image)
base_image = base_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps
).images[0]

# Save the base image
base_image.save("output_base_SDXL1_0.png")

# Refine the generated image by reintroducing it as input (Image-to-Image)
refined_image = refiner_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=30,
    image=base_image,
    strength=0.5
).images[0]

# Save the refined image
refined_image.save("output_refined_SDXL1_0.png")

print("Images generated and saved: 'output_base_image.png' and 'output_refined_image.png'")
