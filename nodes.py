# KSamplerTiled by Paulo Coronado - April 2023
import json
import os
import sys
import time

import comfy.samplers
import folder_paths
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torch import Tensor

from nodes import common_ksampler

from .utils import *

# Global variables
last_returned_ids = {}
my_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))
sys.path.append(comfy_dir)
font_path = os.path.join(my_dir, 'arial.ttf')
MAX_RESOLUTION = 1024


class KSamplerTiled:
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    def __init__(self):
        self.output_dir = os.path.join(comfy_dir, 'temp')
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "scale": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            "controlnet_tile_model": (folder_paths.get_filename_list("controlnet"), ),
            "controlnet_pyrUp": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            "concurrent_tiles": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "preview_image": (["Enabled", "Disabled"],),
        },
            "optional": {"optional_vae": ("VAE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",
                    "LATENT", "VAE", "IMAGE")
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-",
                    "LATENT", "VAE", "IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, scale, controlnet_tile_model, controlnet_pyrUp, concurrent_tiles, denoise, preview_image, optional_vae=(None,), prompt=None, extra_pnginfo=None):

        # Functions for previewing images in Ksampler
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(input):
            input = input.replace("%width%", str(images[0].shape[1]))
            input = input.replace("%height%", str(images[0].shape[0]))
            return input

        def preview_images(images, filename_prefix):
            filename_prefix = compute_vars(filename_prefix)

            subfolder = os.path.dirname(os.path.normpath(filename_prefix))
            filename = os.path.basename(os.path.normpath(filename_prefix))

            full_output_folder = os.path.join(self.output_dir, subfolder)

            try:
                counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                     map(map_filename, os.listdir(full_output_folder))))[0] + 1
            except ValueError:
                counter = 1
            except FileNotFoundError:
                os.makedirs(full_output_folder, exist_ok=True)
                counter = 1

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            results = list()
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                file = f"{filename}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file),
                         pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
            return results

        filename_prefix = "KST_{:02d}".format(int(time.time()))

        output_tiles = []
        vae = optional_vae

        if vae == (None,):
            preview_image = "Disabled"

        # Initialize latent
        latent: Tensor | None = None

        # Split latent image in tiles
        tiles = get_tiles(latent_image, scale)

        # For each tile: decode, upscale, apply controlnet and ksampler
        for i, tile in enumerate(tiles):
            # Decode, upscale and encode
            tile_image = decode(vae, tile)
            upscaled_image = upscale_image(tile_image, scale)

            # Apply controlnet
            positive_control = apply_controlnet_tile(
                positive, controlnet_tile_model, tile_image, controlnet_pyrUp)

            # Sample using KSampler
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive_control, negative, {
                                      'samples': encode(vae, upscaled_image)}, denoise=denoise)

            output_tiles.append(samples[0]["samples"])

        # Merge the latent samples into a single tensor
        latent = merge_images(output_tiles)

        if preview_image == "Disabled":
            return {"ui": {"images": list()}, "result": (model, positive, negative, {"samples": latent}, vae, KSamplerTiled.empty_image,)}
        else:
            images = vae.decode(latent).cpu()
            results = preview_images(images, filename_prefix)
            return {"ui": {"images": results}, "result": (model, positive, negative, {"samples": latent}, vae, images,)}


NODE_CLASS_MAPPINGS = {
    "KSamplerTiled": KSamplerTiled,
}
