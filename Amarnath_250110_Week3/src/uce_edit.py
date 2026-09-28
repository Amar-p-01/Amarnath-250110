# Cell 2 — write the complete UCE editor
'''
src/uce_edit.py

Offline Unified Concept Editing (UCE) preparation for Picture Imperfect.
Run this from the repository root. It writes a standard Diffusers UNet to
modified_weights/unet/. The graded runtime only needs to load that folder.

The UCE mathematics is unchanged:
    W_new = W_old + (V_target - W_old K) K.T
                         @ pinv(K K.T + lambda C_0)

The only execution change is that independent attention layers are sharded
between the available CUDA devices. The base pipeline and text encoder stay
on CPU except while prompt embeddings are being built, so the resulting
weights remain compatible with a single-T4 grading runtime.
'''

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = 'CompVis/stable-diffusion-v1-4'
REVISION = '133a221'  # pinned -- do not change
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'modified_weights',
)
LAMBDA_REG = 0.005


def eos(embedding_batch):
    '''Take only the last (EOS/pooled) token position per prompt.'''
    return embedding_batch[:, -1:, :]


RESTRICTED_STYLES = {
    'Wassily Kandinsky': [
        'a painting by Wassily Kandinsky',
        'art in Kandinsky style',
        'an abstract geometric composition with bold colors like Kandinsky',
        'a non-representational painting with circles and lines like Kandinsky',
        'a painting by Wassily Kandinsky',
    ],
    'Gustav Klimt': [
        'a painting by Gustav Klimt',
        'golden mosaic art by Klimt',
        'a painting with gold leaf ornamentation by Klimt',
        'a painting by Gustav Klimt',
        'golden mosaic art by Klimt',
    ],
    'Edvard Munch': [
        'a painting by Edvard Munch',
        'expressionist artwork by Munch',
        'a painting by Edvard Munch',
    ],
    'Frida Kahlo': [
        'a painting by Frida Kahlo',
        'self portrait by Frida Kahlo',
        'a painting by Frida Kahlo',
        'self portrait by Frida Kahlo',
        'a painting by Frida Kahlo',
    ],
    'Amedeo Modigliani': [
        'a painting by Amedeo Modigliani',
        'art by Modigliani',
        'a painting by Amedeo Modigliani',
    ],
}

NEUTRAL_PROMPTS = [
    'a high quality photo',
    'a detailed image',
    'a landscape photograph',
    'a simple abstract pattern',
]

DEBIAS_OCCUPATIONS = [
    ('a photo of a CEO', 'a photo of a male CEO', 'a photo of a female CEO', 0.5),
    ('a photo of a Chef', 'a photo of a male Chef', 'a photo of a female Chef', 0.5),
]

DEBIAS_OCCUPATIONS_OVERSHOOT = [
    ('a photo of a Nurse', 'a photo of a male Nurse', 'a photo of a female Nurse', 0.62),
    (
        'a photo of a Construction Worker',
        'a photo of a female Construction Worker',
        'a photo of a male Construction Worker',
        0.62,
    ),
]

SELF_PRESERVE_ROLES = ['a photo of a Secretary'] * 16

PRESERVE_PROMPTS = [
    'a photo of a mountain landscape at sunset',
    'a still life of fruit on a wooden table',
    'a photo of a golden retriever running on a beach',
    'a photo of a firefighter standing in front of a truck',
    'an illustration of a cozy cabin in the woods',
]


def encode_prompts(tokenizer, text_encoder, prompts, device):
    tokens = tokenizer(
        prompts,
        padding='max_length',
        max_length=77,
        return_tensors='pt',
        truncation=True,
    ).input_ids.to(device)
    with torch.no_grad():
        embeddings = text_encoder(tokens)[0]
    return embeddings.detach().float().cpu()


def build_edit_matrices(tokenizer, text_encoder, device):
    '''Build the exact joint K and V matrices used by the original UCE edit.'''
    preserve_embs = encode_prompts(
        tokenizer, text_encoder, PRESERVE_PROMPTS, device
    )
    e_preserve = preserve_embs.view(-1, 768).T
    c_0 = torch.matmul(e_preserve, e_preserve.T)

    neutral_embs = encode_prompts(
        tokenizer, text_encoder, NEUTRAL_PROMPTS, device
    ).mean(dim=0, keepdim=True)

    k_list = []
    v_target_ref_list = []

    # Style erasure: full-token style prompts -> neutral full-token target.
    for variations in RESTRICTED_STYLES.values():
        style_emb = encode_prompts(tokenizer, text_encoder, variations, device)
        k_list.append(style_emb.view(-1, 768).T)
        neutral_target = neutral_embs.repeat(len(variations), 1, 1)
        v_target_ref_list.append(neutral_target.view(-1, 768).T)

    # CEO/Chef: full-token interpolation between the original male/female mappings.
    for neutral_prompt, male_prompt, female_prompt, push in DEBIAS_OCCUPATIONS:
        k_neut = encode_prompts(tokenizer, text_encoder, [neutral_prompt], device).squeeze(0)
        k_male = encode_prompts(tokenizer, text_encoder, [male_prompt], device).squeeze(0)
        k_fem = encode_prompts(tokenizer, text_encoder, [female_prompt], device).squeeze(0)
        k_list.append(k_neut.T)
        v_target_ref_list.append((push * k_male + (1.0 - push) * k_fem).T)

    # Nurse/Construction Worker: EOS-only mild overshoot.
    for neutral_prompt, toward_prompt, away_prompt, push in DEBIAS_OCCUPATIONS_OVERSHOOT:
        k_neut = eos(encode_prompts(tokenizer, text_encoder, [neutral_prompt], device)).squeeze(0)
        k_toward = eos(encode_prompts(tokenizer, text_encoder, [toward_prompt], device)).squeeze(0)
        k_away = eos(encode_prompts(tokenizer, text_encoder, [away_prompt], device)).squeeze(0)
        k_list.append(k_neut.T)
        v_target_ref_list.append((push * k_toward + (1.0 - push) * k_away).T)

    # Secretary: repeated EOS self-preservation target.
    for prompt in SELF_PRESERVE_ROLES:
        k_self = eos(encode_prompts(tokenizer, text_encoder, [prompt], device)).squeeze(0)
        k_list.append(k_self.T)
        v_target_ref_list.append(k_self.T)

    k_target = torch.cat(k_list, dim=1)
    v_target_ref = torch.cat(v_target_ref_list, dim=1)
    if k_target.shape[1] != v_target_ref.shape[1]:
        raise RuntimeError(
            f'K/V column mismatch: {k_target.shape} vs {v_target_ref.shape}'
        )
    return k_target, v_target_ref, c_0


def attention_layers(unet):
    return [
        (name, module)
        for name, module in unet.named_modules()
        if 'attn2' in name and ('to_k' in name or 'to_v' in name)
    ]


def _edit_gpu_shard(gpu_id, layers, k_target, v_target_ref, inv_cov):
    '''Edit one disjoint layer shard on one GPU.'''
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    k = k_target.to(device)
    v_ref = v_target_ref.to(device)
    inverse = inv_cov.to(device)

    with torch.no_grad():
        for name, module in layers:
            w_old = module.weight.detach().to(device)
            v_target = torch.matmul(w_old, v_ref)
            residual = v_target - torch.matmul(w_old, k)
            delta_w = torch.matmul(
                torch.matmul(residual, k.T), inverse
            )
            w_new_cpu = (w_old + delta_w).to('cpu')
            module.weight.copy_(w_new_cpu)
            del w_old, v_target, residual, delta_w, w_new_cpu

    torch.cuda.synchronize(gpu_id)
    return gpu_id, len(layers)


def apply_sharded_uce(unet, k_target, v_target_ref, c_0):
    layers = attention_layers(unet)
    if not layers:
        raise RuntimeError('No UNet attn2.to_k/to_v layers were found.')

    # This inverse depends only on the shared K and preservation covariance,
    # so computing it once is algebraically identical to recomputing it per layer.
    cov = torch.matmul(k_target, k_target.T) + LAMBDA_REG * c_0
    inv_cov = torch.linalg.pinv(cov)

    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(min(gpu_count, 2)))
    if not gpu_ids:
        print('CUDA unavailable; applying the same UCE solve on CPU.')
        with torch.no_grad():
            for name, module in layers:
                w_old = module.weight.detach()
                v_target = torch.matmul(w_old, v_target_ref)
                residual = v_target - torch.matmul(w_old, k_target)
                delta_w = torch.matmul(
                    torch.matmul(residual, k_target.T), inv_cov
                )
                module.weight.copy_(w_old + delta_w)
        return len(layers)

    shards = [[] for _ in gpu_ids]
    for index, item in enumerate(layers):
        shards[index % len(gpu_ids)].append(item)
    for gpu_id, shard in zip(gpu_ids, shards):
        print(f'cuda:{gpu_id} receives {len(shard)} attention layers')

    # Each worker owns disjoint CPU modules and one CUDA device.
    # Both T4s therefore process layer shards concurrently.
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(
                _edit_gpu_shard,
                gpu_id,
                shard,
                k_target,
                v_target_ref,
                inv_cov,
            )
            for gpu_id, shard in zip(gpu_ids, shards)
        ]
        for future in futures:
            future.result()

    del cov, inv_cov
    for gpu_id in gpu_ids:
        torch.cuda.synchronize(gpu_id)
    torch.cuda.empty_cache()
    return len(layers)


def run_uce_edit():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'Loading {MODEL_ID} at revision {REVISION} on CPU...')
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float32,
    )

    # The text encoder is used only to construct the offline UCE matrices.
    # It is not part of the saved edit and the safety checker is untouched.
    embedding_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipe.text_encoder.to(embedding_device)
    k_target, v_target_ref, c_0 = build_edit_matrices(
        pipe.tokenizer, pipe.text_encoder, embedding_device
    )
    pipe.text_encoder.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f'K_target: {tuple(k_target.shape)}, '
        f'V_target_ref: {tuple(v_target_ref.shape)}'
    )
    edited_count = apply_sharded_uce(
        pipe.unet, k_target, v_target_ref, c_0
    )
    print(f'Edited {edited_count} attn2.to_k/to_v layers.')

    output_unet = os.path.join(OUTPUT_DIR, 'unet')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_pretrained(output_unet, safe_serialization=True)
    print(f'Saved edited UNet to {output_unet}')
    print('No runtime prompt branching, filtering, or rewriting was added.')


if __name__ == '__main__':
    run_uce_edit()
