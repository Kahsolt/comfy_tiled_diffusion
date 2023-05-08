# ComfyUI Custom Nodes Introduction

## How to create a custom node

Developing ComfyUI custom nodes is straightforward. Follow these steps to create your custom node:

1. Create a new folder under `ComfyUI/custom_nodes`;
2. Inside the folder you created, add a file named `__init__.py`. This file will point to your node function.

Example of `__init__.py` content:
```python
from .nodes import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
```
3. Create a file called `nodes.py` in the same folder. This file will contain the structure of your node.

Example of `nodes.py` content:
```python
class CustomNodeName:
  def __init__(self):
        self.output_dir = os.path.join(comfy_dir, 'temp')
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
          "model": ("MODEL",),
          "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
          "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
          # add more fields...
        }

    RETURN_TYPES = ("LATENT")
    RETURN_NAMES = ("LATENT")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
  # This is the main function
  def sample(self, model, seed, steps):
    # Your code logic goes here...
    return {"ui": {"images": results}, "result": ({"samples": latent})}

NODE_CLASS_MAPPINGS = {
    "CustomNodeName": CustomNodeName,
}
```
Make sure to replace "CustomNodeName" with the actual name you want to give to your custom node.

Feel free to add more fields and customize the node according to your requirements.
