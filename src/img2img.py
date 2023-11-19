from pathlib import Path
import random
import torch
from diffusers import StableDiffusionImg2ImgPipeline


def img2img(base_model_filepath, image, prompt, negative_prompt=None, strength=0.4, guidance_scale=7, scheduler=False, clip_skip=2, seed=None, safety_checker=True, lora_model_filepath=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img
    pipeline = StableDiffusionImg2ImgPipeline.from_single_file(base_model_filepath)

    # load lora model (https://huggingface.co/docs/diffusers/main/en/training/lora#dreambooth-inference)
    if lora_model_filepath:
        lora_model_filepath = Path(lora_model_filepath)
        pipeline.load_lora_weights(lora_model_filepath.parent, weight_name=lora_model_filepath.name)

    # set schedulers (https://huggingface.co/docs/diffusers/api/schedulers/overview)
    if scheduler:
        if scheduler == Scheduler.DPM_2M_Karras:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        if scheduler == Scheduler.DPM_SDE_Karras:
            pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        if scheduler == Scheduler.Euler_a:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # set seed value
    if seed:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = torch.Generator(device).manual_seed(random.randint(0, 2**32))

    # deactivate safety checker
    if not safety_checker:
        def null_safety(images, **kwargs):
            return images, [False]
        pipeline.safety_checker = null_safety

    pipeline = pipeline.to(device)

    i2i_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        clip_skip=clip_skip,
    ).images[0]

    return i2i_image
