import random
import torch
from diffusers import StableDiffusionLatentUpscalePipeline

# https://note.com/npaka/n/n75fcd514d57a
def latent_upscaler(image, prompt, negative_prompt=None, guidance_scale=7, num_inference_steps=10, seed=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # setup upscaler
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
    )

    # set seed value
    if seed:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = torch.Generator(device).manual_seed(random.randint(0, 2**32))
    
    upscaler.to(device)

    upscaled_image = upscaler(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    return upscaled_image
