import sys
from typing import *

from torch import Tensor

from .extra_networks import ExtraNetworkParams
from .prompt_parser import (MulticondLearnedConditioning,
                            ScheduledPromptConditioning)
from .sd_samplers_compvis import VanillaStableDiffusionSampler
from .sd_samplers_kdiffusion import KDiffusionSampler

ModuleType = type(sys)

Sampler = Union[KDiffusionSampler, VanillaStableDiffusionSampler]
Cond = MulticondLearnedConditioning
Uncond = List[List[ScheduledPromptConditioning]]
ExtraNetworkData = DefaultDict[str, List[ExtraNetworkParams]]

# 'c_crossattn': Tensor    # prompt cond
# 'c_concat':    Tensor    # latent mask
CondDict = Dict[str, Tensor]
