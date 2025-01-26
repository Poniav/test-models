from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

# Load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
path = os.path.abspath("resources/marlin.jpeg")
print(path)

# Load the input image
input_image = Image.open(path).convert("RGB")
input_image = input_image.resize((512, 512))

# Generate the results
prompts = ["similar composition, faithful to the original, realistic, detailed",]
results = []

for prompt in prompts:
    images = pipe(prompt=prompt, image=input_image, strength=0.2, guidance_scale=7.5).images
    results.extend(images)

# Save the results
for i, img in enumerate(results):
    img.save(f"result_{i}.png")