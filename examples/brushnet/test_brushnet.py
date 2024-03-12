from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image

base_model_path = "runwayml/stable-diffusion-v1-5"
brushnet_path = "ckpt_path"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

image_path="examples/brushnet/src/test_image.jpg"
mask_path="examples/brushnet/src/test_mask.jpg"
caption="A cake on the table."

init_image = cv2.imread(image_path)
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(1234)

image = pipe(
    caption, 
    init_image, 
    mask_image, 
    num_inference_steps=50, 
    generator=generator,
    paintingnet_conditioning_scale=1.0
).images[0]
image.save("output.png")