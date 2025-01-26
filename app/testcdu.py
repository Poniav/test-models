import torch
from diffusers import StableDiffusionPipeline

# Vérifier la présence d'un GPU compatible
if torch.cuda.is_available():
    print("CUDA disponible : GPU détecté")
else:
    print("CUDA indisponible : exécution sur CPU")

# Charger le pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

print("Pipeline successfully loaded!")

# Exemple d'utilisation

