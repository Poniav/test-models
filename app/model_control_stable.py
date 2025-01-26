from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import torch
from config import Config

# Load the configuration
app = Config()

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")

# Load the Stable Diffusion pipeline
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Load the input image
from PIL import Image
import numpy as np
import cv2

path = app.RESOURCES_PATH + "/marlin.jpeg"
print(path)
input_image = Image.open(path).convert("RGB")
input_array = np.array(input_image)
edges = cv2.Canny(input_array, 100, 200)  # Apply Canny edge detection
control_image = Image.fromarray(edges).convert("RGB")

# Generate the result
results = pipe(
    prompt="A beautiful portrait with similar style",
    image=input_image,
    control_image=control_image,
    strength=0.5,
    guidance_scale=7.5
).images

# Save the result
img_path = app.RESULTS_PATH + "/result_fidele.png"
results[0].save(img_path)
