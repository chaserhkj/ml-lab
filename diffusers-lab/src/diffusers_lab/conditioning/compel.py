from compel import CompelForSDXL
from diffusers import StableDiffusionXLPipeline
from typing import Any
from torch import Tensor

def get_sdxl_prompt_embeds(pipe: StableDiffusionXLPipeline, pos: str, neg: str) -> dict[str, Tensor | None]:
    compel = CompelForSDXL(pipe)
    conds = compel([pos, neg])
    pos_embed = conds.embeds[0:1]
    neg_embed = conds.embeds[1:2]
    pos_pooled_embed = None if conds.pooled_embeds is None else conds.pooled_embeds[0:1]
    neg_pooled_embed = None if conds.pooled_embeds is None else conds.pooled_embeds[1:2]
    return {
        "prompt_embeds": pos_embed,
        "pooled_prompt_embeds": pos_pooled_embed,
        "negative_prompt_embeds": neg_embed,
        "negative_pooled_prompt_embeds": neg_pooled_embed
    }
