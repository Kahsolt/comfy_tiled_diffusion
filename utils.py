import math
import os

import cv2
import numpy as np
import torch
from PIL import Image

import comfy.samplers
import comfy.utils
import folder_paths
from comfy import model_management
from comfy_extras.chainner_models import model_loading

from .tiled_diffusion.sample import prepare_noise, sample

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")

""" Image """

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    mask_list = [x.squeeze().numpy().astype(np.uint8)
                 for x in torch.split(img_tensor, 1)]
    return mask_list

def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = 0
    if resolution is not None:
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H),
                     interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def upscale_image(image, scale):
    samples = image.movedim(-1, 1)

    width = samples.shape[3] * scale
    height = samples.shape[2] * scale

    s = comfy.utils.common_upscale(samples, width, height, 'area', 'center')
    s = s.movedim(1, -1)
    upscaled_image = s

    return upscaled_image

""" Models """

def update_loaded_objects(prompt):
    global loaded_objects

    # Extract all Efficient Loader class type entries
    efficient_loader_entries = [entry for entry in prompt.values(
    ) if entry["class_type"] == "Efficient Loader"]

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
        unused_indices = [i for i, entry in enumerate(
            loaded_objects[list_key]) if entry[0] not in desired_ckpt_names]
        for index in sorted(unused_indices, reverse=True):
            loaded_objects[list_key].pop(index)

    # Check and clear unused vae entries
    unused_vae_indices = [i for i, entry in enumerate(
        loaded_objects["vae"]) if entry[0] not in desired_vae_names]
    for index in sorted(unused_vae_indices, reverse=True):
        loaded_objects["vae"].pop(index)

    # Check and clear unused lora entries
    unused_lora_indices = [i for i, entry in enumerate(
        loaded_objects["lora"]) if entry[0] not in desired_lora_names]
    for index in sorted(unused_lora_indices, reverse=True):
        loaded_objects["lora"].pop(index)

def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path)
    out = model_loading.load_state_dict(sd).eval()
    return out

""" Models (VAE) """

def load_vae(vae_name):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    """
    global loaded_objects

    # Check if vae_name exists in "vae" array
    if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
        # Extract the second tuple entry of the checkpoint
        vae = [entry[1]
               for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        # Update loaded_objects[] array
        loaded_objects["vae"].append((vae_name, vae))
    return vae

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
    t = vae.encode(pixels[:, :, :, :3])
    return t

def decode(vae, samples):
    return vae.decode(samples)

""" Sampling """

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
            match = all(inputs[param_name] == param_value for param_name,
                        param_value in input_params if param_value is not None)

            if match:
                matching_ids.append(key)

    if matching_ids:
        input_key = tuple(param_value for param_name,
                          param_value in input_params)

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

def load_vae(vae_name):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    """
    global loaded_objects

    # Check if vae_name exists in "vae" array
    if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
        # Extract the second tuple entry of the checkpoint
        vae = [entry[1]
               for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        # Update loaded_objects[] array
        loaded_objects["vae"].append((vae_name, vae))
    return vae

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
    t = vae.encode(pixels[:, :, :, :3])
    return t

def decode(vae, samples):
    return vae.decode(samples["samples"])

def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path)
    out = model_loading.load_state_dict(sd).eval()
    return out

def upscale_image(image, upscale_model):
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=128 + 64, tile_y=128 + 64, overlap = 8, upscale_amount=upscale_model.scale)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return s

def scale_image(original, upscaled, scale):
    original_samples = original.movedim(-1, 1)
    upscaled_samples = upscaled.movedim(-1, 1)

    width = int(original_samples.shape[3] * scale)
    height = int(original_samples.shape[2] * scale)

    s = comfy.utils.common_upscale(upscaled_samples, width, height, 'area', 'center')
    s = s.movedim(1, -1)

    return s

def get_tiles(latent, scale):
    """
    Splits a latent vector into a list of tiles with the given tile height and width.
    If the image dimensions are not evenly divisible by the tile size, adds latent noise padding.
    """
    latent = latent['samples']

    # Determine the dimensions of the input image
    _, _, height, width = latent.shape

    if (height % scale != 0) or (width % scale != 0):
        raise ValueError(
            'Image dimensions must be evenly divisible by the scale factor')

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

def merge_images(tiles, overlap:int=64):
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
            image[:, :, i*height:(i+1)*height, j*width:(j+1)
                  * width] = tiles[i*scale + j]

    return image

""" ControlNet """

def load_controlnet(control_net_name):
    controlnet_path = folder_paths.get_full_path(
        "controlnet", control_net_name)
    controlnet = comfy.sd.load_controlnet(controlnet_path)
    return controlnet

def apply_controlnet_tile(conditioning, controlnet, image, controlnet_pyrUp=3, strength=1.0):
    # Load controlnet
    controlnet = load_controlnet(controlnet)

    # Apply TilePreprocessor
    np_detected_map = common_annotator_call(
        preprocess, image, controlnet_pyrUp)
    processed_image = (img_np_to_tensor(np_detected_map),)

    c = []
    control_hint = processed_image[0].movedim(-1, 1)
    print(control_hint.shape)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = controlnet.copy().set_cond_hint(control_hint, strength)
        if 'control' in t[1]:
            c_net.set_previous_controlnet(t[1]['control'])
        n[1]['control'] = c_net
        c.append(n)
    return c

def preprocess(image, pyrUp_iters=3):
    H, W, C = image.shape
    detected_map = cv2.resize(image, (W // (2 ** pyrUp_iters),
                              H // (2 ** pyrUp_iters)), interpolation=cv2.INTER_AREA)
    for _ in range(pyrUp_iters):
        detected_map = cv2.pyrUp(detected_map)
    return detected_map

def common_annotator_call(annotator_callback, tensor_image, *args):
    tensor_image_list = img_tensor_to_np(tensor_image)
    out_list = []
    for tensor_image in tensor_image_list:
        call_result = annotator_callback(
            resize_image(HWC3(tensor_image)), *args)
        H, W, C = tensor_image.shape
        out_list.append(cv2.resize(HWC3(call_result), (W, H),
                        interpolation=cv2.INTER_AREA))
    return out_list

""" MultiDiffusion """

def multi_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        skip = latent["batch_index"] if "batch_index" in latent else 0
        noise = prepare_noise(latent_image, seed, skip)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x):
        pbar.update_absolute(step + 1)

    samples = sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback)
    out = latent.copy()
    out["samples"] = samples
    return (out, )