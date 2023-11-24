from logging import getLogger, Formatter, StreamHandler, INFO
import random
import torch


formatter = Formatter("【%(levelname)s】%(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(INFO)


def set_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def initialize_generator(seed):
    device = set_device()
    if seed:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = torch.Generator(device).manual_seed(random.randint(0, 2**32))

    return generator


def deactivate_safety_checker(pipeline):
    def null_safety(images, **kwargs):
        return images, [False]
    pipeline.safety_checker = null_safety

    logger.info("safety checker has been deactivated. NSFW image may be output.")
    return pipeline