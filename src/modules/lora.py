from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path

formatter = Formatter("【%(levelname)s】%(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(INFO)


class Lora:
    def __init__(self, filepath, scale=1.0):
        self.__filepath = Path(filepath)
        self.__scale = scale

    def get_filepath(self):
        return self.__filepath

    def get_scale(self):
        return self.__scale


# Lora (https://huggingface.co/docs/diffusers/training/lora)
def set_lora(pipeline, lora:Lora):
    lora_model_filepath = lora.get_filepath()
    pipeline.load_lora_weights(lora_model_filepath.parent, weight_name=lora_model_filepath.name)
    pipeline.fuse_lora(lora_scale=lora.get_scale())

    logger.info(f"load lora model : {lora_model_filepath}")
    return pipeline
