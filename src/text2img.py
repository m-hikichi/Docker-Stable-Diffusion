from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
from enum import Enum, unique
import random
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler


formatter = Formatter("【%(levelname)s】%(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(INFO)


@unique
class Scheduler(Enum):
    DPM_2M_Karras = "DPM++ 2M Karras"
    DPM_SDE_Karras = "DPM++ SDE Karras"
    Euler_a = "Euler a"


def text2img(base_model_filepath, prompt, negative_prompt=None, width=512, height=512, guidance_scale=7, num_inference_steps=30, scheduler=False, seed=None, safety_checker=True, lora_model_filepath=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # load base model (https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors)
    pipeline = StableDiffusionPipeline.from_single_file(base_model_filepath)
    logger.info(f"load base model : {base_model_filepath}")

    # load lora model (https://huggingface.co/docs/diffusers/main/en/training/lora#dreambooth-inference)
    if lora_model_filepath:
        lora_model_filepath = Path(lora_model_filepath)
        pipeline.load_lora_weights(lora_model_filepath.parent, weight_name=lora_model_filepath.name)
        logger.info(f"load lora model : {lora_model_filepath}")

    # set schedulers (https://huggingface.co/docs/diffusers/api/schedulers/overview)
    if scheduler:
        if scheduler == Scheduler.DPM_2M_Karras:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
            logger.info(f"set scheduler : {scheduler}")
        if scheduler == Scheduler.DPM_SDE_Karras:
            pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
            logger.info(f"set scheduler : {scheduler}")
        if scheduler == Scheduler.Euler_a:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            logger.info(f"set scheduler : {scheduler}")

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
        logger.info("safety checker has been deactivated. NSFW image may be output.")

    pipeline = pipeline.to(device)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return image
