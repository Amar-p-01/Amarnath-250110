# Cell 3 — participant runtime loader

import os

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image

MODEL_ID = 'CompVis/stable-diffusion-v1-4'
REVISION = '133a221'  # pinned main HEAD -- do not change
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32


def load_model():
    '''Return the pipeline with the offline-edited UNet loaded.'''
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=DTYPE,
        # Safety checker remains active: never set it to None.
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=False)

    edited_unet_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'modified_weights',
        'unet',
    )
    edited_unet = UNet2DConditionModel.from_pretrained(
        edited_unet_path,
        torch_dtype=DTYPE,
    )
    pipe.unet = edited_unet.to(DEVICE)
    return pipe


@torch.no_grad()
def generate(pipe, prompt: str, seed: int) -> Image.Image:
    '''Generate one image. Keep this signature unchanged.'''
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    output = pipe(
        prompt,
        num_inference_steps=50,
        generator=generator,
    )
    image = output.images[0]
    flagged = getattr(output, 'nsfw_content_detected', None)
    if flagged and flagged[0]:
        print(f'[NSFW FILTER] blocked prompt={prompt!r} seed={seed}')
    return image
