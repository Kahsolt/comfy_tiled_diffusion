from .sd_samplers_compvis import *
from .sd_samplers_kdiffusion import *

all_samplers = [
    *samplers_data_k_diffusion,
    *samplers_data_compvis,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers = []
samplers_for_img2img = []
samplers_map = {}


def create_sampler(name, model):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    assert config is not None, f'bad sampler name: {name}'

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img

    # hidden = set(shared.opts.hide_samplers)
    # hidden_img2img = set(shared.opts.hide_samplers + ['PLMS', 'UniPC'])
    hidden = []
    hidden_img2img = []

    samplers = [x for x in all_samplers if x.name not in hidden]
    samplers_for_img2img = [x for x in all_samplers if x.name not in hidden_img2img]

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


set_samplers()
