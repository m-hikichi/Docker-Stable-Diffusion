from diffusers import StableDiffusionLatentUpscalePipeline
from modules.util import set_device, initialize_generator

# https://note.com/npaka/n/n75fcd514d57a
def latent_upscaler(image, prompt, negative_prompt=None, guidance_scale=7, num_inference_steps=10, seed=None):
    
    # setup upscaler
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
    )

    # set seed value
    generator = initialize_generator(seed)
    
    # set device
    device = set_device()
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
