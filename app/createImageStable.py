from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import os
import torch

# Charger le modèle ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")

# Charger le pipeline avec ControlNet
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Utiliser ControlNet pour capturer les contours de l'image de départ
from PIL import Image
import numpy as np
import cv2

path = os.path.abspath("resources/aurore.png")
print(path)
input_image = Image.open(path).convert("RGB")
input_array = np.array(input_image)
edges = cv2.Canny(input_array, 100, 200)  # Détecter les contours
control_image = Image.fromarray(edges).convert("RGB")

# Générer une image fidèle aux contours
results = pipe(
    prompt="A beautiful portrait with similar style",
    image=input_image,
    control_image=control_image,
    strength=0.5,
    guidance_scale=7.5
).images

# Sauvegarder l'image
results[0].save("result_fidele.png")
