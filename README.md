# Tiled Diffusion for ComfyUI

> Enhance, Upscale, and Fix Images with Advanced Tiling Techniques

Experience the next level of image enhancement, upscaling, and fixing with the ComfyUI Custom Node. This innovative tool combines three cutting-edge tiling techniques - [ControlNet v1.1 Tile](https://github.com/lllyasviel/ControlNet-v1-1-nightly), [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) and [MultiDiffusion](https://multidiffusion.github.io/) - to transform your images into stunning, high-quality visuals.

## Installation

### Clone the repo

Navigate to the `ComfyUI\custom_nodes` directory in your ComfyUI installation. Open your CLI or terminal and run:

```sh
git clone https://github.com/paulo-coronado/comfy_tiled_diffusion/
```

## Usage example

1. Add the "TiledDiffusionKSampler" node to your ComfyUI workflow. This node serves as the entry point for applying the advanced tiling techniques to your images;
2. Configure the parameters and settings according to your desired image enhancement requirements.

## Release History

### Version 0.1
 - [x] Upscaling with models and custom resizing
 - [ ] MultiDiffusion support - *UNDER CONSTRUCTION* 🛠️
 - [ ] ControlNet Tile 1.1 support
 - [ ] Mixture of Diffusers support

## Contributing

1. Fork it (<https://github.com/paulo-coronado/comfy_tiled_diffusion/>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
