import diffusers
from enum import Enum, unique
from logging import getLogger, Formatter, StreamHandler, INFO


formatter = Formatter("【%(levelname)s】%(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)
logger = getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(INFO)


# Schedulers (https://huggingface.co/docs/diffusers/api/schedulers/overview)
@unique
class Scheduler(Enum):
    DPM_2M = "DPM++ 2M"
    DPM_2M_Karras = "DPM++ 2M Karras"
    DPM_2M_SDE = "DPM++ 2M SDE"
    DPM_2M_SDE_Karras = "DPM++ 2M SED Karras"
    DPM_2S_a = "DPM++ 2S a"
    DPM_2S_a_Karras = "DPM++ 2S a Karras"
    DPM_SDE = "DPM++ SDE"
    DPM_SDE_Karras = "DPM++ SDE Karras"
    DPM2 = "DPM2"
    DPM2_Karras = "DPM2 Karras"
    DPM2_a = "DPM2 a"
    DPM2_a_Karras = "DPM2 a Karras"
    DPM_adaptive = "DPM adaptive"
    DPM_fast = "DPM fast"
    Euler = "Euler"
    Euler_a = "Euler a"
    Heun = "Heun"
    LMS = "LMS"
    LMS_Karras = "LMS Karras"


def set_scheduler(pipeline, scheduler: Scheduler):
    if scheduler:
        if scheduler == Scheduler.DPM_2M:
            pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.DPM_2M_Karras:
            pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        
        if scheduler == Scheduler.DPM_2M_SDE:
            pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")
        
        if scheduler == Scheduler.DPM_2M_SDE_Karras:
            pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
            
        if scheduler == Scheduler.DPM_2S_a:
            pass

        if scheduler == Scheduler.DPM_2S_a_Karras:
            pass

        if scheduler == Scheduler.DPM_SDE:
            pipeline.scheduler = diffusers.DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)

        if scheduler == Scheduler.DPM_SDE_Karras:
            pipeline.scheduler = diffusers.DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        
        if scheduler == Scheduler.DPM2:
            pipeline.scheduler = diffusers.KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.DPM2_Karras:
            pipeline.scheduler = diffusers.KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

        if scheduler == Scheduler.DPM2_a:
            pipeline.scheduler = diffusers.KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.DPM2_a_Karras:
            pipeline.scheduler = diffusers.KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

        if scheduler == Scheduler.DPM_adaptive:
            pass

        if scheduler == Scheduler.DPM_fast:
            pass

        if scheduler == Scheduler.Euler:
            pipeline.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

        if scheduler == Scheduler.Euler_a:
            pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.Heun:
            pipeline.scheduler = diffusers.HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.LMS:
            pipeline.scheduler = diffusers.LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        if scheduler == Scheduler.LMS_Karras:
            pipeline.scheduler = diffusers.LMSDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

        logger.info(f"set scheduler : {scheduler}")
    return pipeline
        
