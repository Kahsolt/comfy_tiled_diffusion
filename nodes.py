# KSamplerTiled by Paulo Coronado - April 2023
import json
import math
import os
import sys
import time

import comfy.samplers
import folder_paths
import numpy as np
import torch
from comfy_extras.chainner_models import model_loading
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torch import Tensor

from nodes import common_ksampler

from .utils import common_annotator_call, img_np_to_tensor, preprocess

last_returned_ids = {}

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

MAX_RESOLUTION=1024

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def load_vae(vae_name):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    """
    global loaded_objects

    # Check if vae_name exists in "vae" array
    if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
        # Extract the second tuple entry of the checkpoint
        vae = [entry[1] for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        # Update loaded_objects[] array
        loaded_objects["vae"].append((vae_name, vae))
    return vae

def update_loaded_objects(prompt):
    global loaded_objects

    # Extract all Efficient Loader class type entries
    efficient_loader_entries = [entry for entry in prompt.values() if entry["class_type"] == "Efficient Loader"]

    # Collect all desired model, vae, and lora names
    desired_ckpt_names = set()
    desired_vae_names = set()
    desired_lora_names = set()
    for entry in efficient_loader_entries:
        desired_ckpt_names.add(entry["inputs"]["ckpt_name"])
        desired_vae_names.add(entry["inputs"]["vae_name"])
        desired_lora_names.add(entry["inputs"]["lora_name"])

    # Check and clear unused ckpt, clip, and bvae entries
    for list_key in ["ckpt", "clip", "bvae"]:
        unused_indices = [i for i, entry in enumerate(loaded_objects[list_key]) if entry[0] not in desired_ckpt_names]
        for index in sorted(unused_indices, reverse=True):
            loaded_objects[list_key].pop(index)

    # Check and clear unused vae entries
    unused_vae_indices = [i for i, entry in enumerate(loaded_objects["vae"]) if entry[0] not in desired_vae_names]
    for index in sorted(unused_vae_indices, reverse=True):
        loaded_objects["vae"].pop(index)

    # Check and clear unused lora entries
    unused_lora_indices = [i for i, entry in enumerate(loaded_objects["lora"]) if entry[0] not in desired_lora_names]
    for index in sorted(unused_lora_indices, reverse=True):
        loaded_objects["lora"].pop(index)

def find_k_sampler_id(prompt, sampler_state=None, seed=None, steps=None, cfg=None,
                      sampler_name=None, scheduler=None, denoise=None, preview_image=None):
    global last_returned_ids

    input_params = [
        ('sampler_state', sampler_state),
        ('seed', seed),
        ('steps', steps),
        ('cfg', cfg),
        ('sampler_name', sampler_name),
        ('scheduler', scheduler),
        ('denoise', denoise),
        ('preview_image', preview_image),
    ]

    matching_ids = []

    for key, value in prompt.items():
        if value.get('class_type') == 'KSamplerTiled':
            inputs = value['inputs']
            match = all(inputs[param_name] == param_value for param_name, param_value in input_params if param_value is not None)

            if match:
                matching_ids.append(key)

    if matching_ids:
        input_key = tuple(param_value for param_name, param_value in input_params)

        if input_key in last_returned_ids:
            last_id = last_returned_ids[input_key]
            next_id = None
            for id in matching_ids:
                if id > last_id:
                    if next_id is None or id < next_id:
                        next_id = id

            if next_id is None:
                # All IDs have been used; start again from the first one
                next_id = min(matching_ids)

        else:
            next_id = min(matching_ids)

        last_returned_ids[input_key] = next_id
        return next_id
    else:
        last_returned_ids.clear()
        return None

def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels

def encode(vae, pixels):
    pixels = vae_encode_crop_pixels(pixels)
    t = vae.encode(pixels[:,:,:,:3])
    return t

def decode(vae, samples):
    return vae.decode(samples)

def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path)
    out = model_loading.load_state_dict(sd).eval()
    return out

def upscale_image(image, scale):
    samples = image.movedim(-1,1)

    width = samples.shape[3] * scale
    height = samples.shape[2] * scale

    s = comfy.utils.common_upscale(samples, width, height, 'area', 'center')
    s = s.movedim(1,-1)
    upscaled_image = s

    return upscaled_image

def get_tiles(latent, scale):
    """
    Splits a latent vector into a list of tiles with the given tile height and width.
    If the image dimensions are not evenly divisible by the tile size, adds latent noise padding.
    """
    latent = latent['samples']

    # Determine the dimensions of the input image
    _, _, height, width = latent.shape

    if (height % scale != 0) or (width % scale != 0):
        raise ValueError('Image dimensions must be evenly divisible by the scale factor')
    
    tile_height = height // scale
    tile_width = width // scale

    print('Tile height: ' + str(tile_height))
    print('Tile width: ' + str(tile_width))

    # Split the image into tiles
    tiles = []
    for i in range(0, scale * tile_height, tile_height):
        for j in range(0, scale * tile_width, tile_width):
            tiles.append(latent[:, :, i:i+tile_height, j:j+tile_width])

    print('Split image into {} tiles'.format(len(tiles)))
    return tiles

def merge_images(tiles):
    """
    Merges a list of tiles into a single image.
    """
    # Determine the dimensions of the output image
    num_tiles = len(tiles)
    _, channels, height, width = tiles[0].shape
    scale = int(math.sqrt(num_tiles))

    # Merge the tiles into a single image
    image = torch.zeros((1, channels, scale * height, scale * width))
    for i in range(scale):
        for j in range(scale):
            image[:, :, i*height:(i+1)*height, j*width:(j+1)*width] = tiles[i*scale + j]

    return image

def load_controlnet(control_net_name):
    controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
    controlnet = comfy.sd.load_controlnet(controlnet_path)
    return controlnet
    
def apply_controlnet_tile(conditioning, controlnet, image, controlnet_pyrUp=3, strength=1.0):
    # Load controlnet
    controlnet = load_controlnet(controlnet)

    # Apply TilePreprocessor
    np_detected_map = common_annotator_call(preprocess, image, controlnet_pyrUp)
    processed_image = (img_np_to_tensor(np_detected_map),)

    c = []
    control_hint = processed_image[0].movedim(-1,1)
    print(control_hint.shape)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = controlnet.copy().set_cond_hint(control_hint, strength)
        if 'control' in t[1]:
            c_net.set_previous_controlnet(t[1]['control'])
        n[1]['control'] = c_net
        c.append(n)
    return c

class KSamplerTiled:
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    def __init__(self):
        self.output_dir = os.path.join(comfy_dir, 'temp')
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
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
            "optional": { "optional_vae": ("VAE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Efficiency Nodes/Sampling"
    
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
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                });
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
            positive_control = apply_controlnet_tile(positive, controlnet_tile_model, tile_image, controlnet_pyrUp)

            # Sample using KSampler
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive_control, negative, {'samples': encode(vae, upscaled_image)}, denoise=denoise)

            output_tiles.append(samples[0]["samples"])

        # Merge the latent samples into a single tensor
        latent = merge_images(output_tiles)

        if preview_image == "Disabled":
            return { "ui": {"images": list()},"result": (model, positive, negative, {"samples": latent}, vae, KSamplerTiled.empty_image,) }
        else:
            images = vae.decode(latent).cpu()
            results = preview_images(images, filename_prefix)
            return {"ui": {"images": results}, "result": (model, positive, negative, {"samples": latent}, vae, images,)}

NODE_CLASS_MAPPINGS = {
    "KSamplerTiled": KSamplerTiled,
}