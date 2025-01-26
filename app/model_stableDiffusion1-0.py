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

# Charger le modèle base SDXL
base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

# Charger le modèle refiner SDXL
refiner_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
).to("cuda")

# Paramètres de génération
prompt = "A futuristic cityscape at sunset, ultra-realistic, vibrant colors"
negative_prompt = "low quality, blurry, unrealistic"
guidance_scale = 7.5  # Poids pour la guidance
num_inference_steps = 50  # Nombre d'étapes d'inférence

# Générer l'image de base
base_image = base_pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                       num_inference_steps=num_inference_steps).images[0]

# Enregistrer l'image générée
base_image.save("output_base_image.png")

# Utiliser le refiner pour améliorer l'image
refined_image = refiner_pipe(image=base_image, prompt=prompt, negative_prompt=negative_prompt,
                             guidance_scale=guidance_scale).images[0]

# Enregistrer l'image affinée
refined_image.save("output_refined_image.png")

print("Images générées et enregistrées : 'output_base_image.png' et 'output_refined_image.png'")
