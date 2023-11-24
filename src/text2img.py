from logging import getLogger, Formatter, StreamHandler, INFO
from diffusers import StableDiffusionPipeline
from modules.util import set_device, initialize_generator, deactivate_safety_checker
from modules.scheduler import set_scheduler
from modules.lora import set_lora


formatter = Formatter("【%(levelname)s】%(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(INFO)


def text2img(base_model_filepath, prompt, negative_prompt=None, width=512, height=512, guidance_scale=7, num_inference_steps=30, scheduler=False, seed=None, safety_checker=True, lora_list=None):
    
    # load base model (https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors)
    pipeline = StableDiffusionPipeline.from_single_file(base_model_filepath)
    logger.info(f"load base model : {base_model_filepath}")

    # set scheduler
    pipeline = set_scheduler(pipeline, scheduler)

    # set seed value
    generator = initialize_generator(seed)

    # deactivate safety checker
    if not safety_checker:
        pipeline = deactivate_safety_checker(pipeline)

    # set lora
    if lora_list:
        for lora in lora_list:
            pipeline = set_lora(pipeline, lora)

    # set device
    device = set_device()
    if device == "cuda":
        logger.info(f"CUDA is available. The device has been set to {device}.")
    elif device == "cpu":
        logger.info(f"CUDA is not available. The device has been set to {device}.")
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
