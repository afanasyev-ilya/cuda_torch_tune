import torch
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:0"

# Instantiate Stable Diffusion Pipeline with FP16 weights
model = DiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
model = model.to(device)

# Optimize the UNet portion with Torch-TensorRT
model.unet = torch.compile(
    model.unet,
    dynamic=False,
)

prompt = "some real cat with gun and other weapons"
image = model(prompt).images[0]

image.save("cat.png")
