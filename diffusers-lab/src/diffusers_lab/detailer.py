import torch
import numpy as np
from torchvision.ops import masks_to_boxes # pyright: ignore[reportMissingTypeStubs]
from diffusers import StableDiffusionXLPipeline # pyright: ignore[reportPrivateImportUsage]
from diffusers import StableDiffusionXLInpaintPipeline # pyright: ignore[reportPrivateImportUsage]
from transformers import Sam3Processor, Sam3Model, logging
from PIL import Image, ImageFilter
from typing import Any

from . import scale_largest_dimension_to
from .conditioning import get_sdxl_prompt_embeds
# pyright: reportUnknownMemberType=none, reportAny=none, reportExplicitAny=none
# pyright: reportUnknownArgumentType=none, reportUnknownVariableType=none

logging.disable_progress_bar()

class SDXLDetailer(object):
    def __init__(self,
            pipeline: StableDiffusionXLPipeline,
            segment_prompt: str,
            segment_threshold: float = 0.5) -> None:
        self._pipeline: StableDiffusionXLPipeline = pipeline
        self._segment_prompt: str = segment_prompt
        self._segment_threshold: float = segment_threshold
    def __run__(self,
                image: Image.Image,
                pos_prompt: str, neg_prompt: str,
                generator: None | torch.Generator = None,
                context_ratio: float = 0.25,
                merge_mask: bool = False,
                num_inference_steps: int = 20,
                guidance_scale: float = 2.5,
                strength: float = 0.4,
                additional_diffusion_params: dict[str, Any] | None = None,
                soft_blend_radius: int = 3) -> Image.Image:
        model = Sam3Model.from_pretrained("facebook/sam3")
        model = model.to("cuda") # pyright: ignore[reportArgumentType]
        processor = Sam3Processor.from_pretrained("facebook/sam3")

        inputs = processor(image, text=self._segment_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        semantic_results = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=inputs["original_sizes"].tolist(), #pyright: ignore[reportAttributeAccessIssue]
            threshold=self._segment_threshold
        )
        masks = torch.stack(semantic_results)
        print(f"Detailer: Segmented {len(masks)} masks")
        if merge_mask:
            print("Detailer: Merging all masks")
            mask_combined = torch.any(masks, dim=0).cpu()
            masks = mask_combined.unsqueeze(0)

        detailed = image.copy()
        for mask in masks:
            detailed_patch, blend_mask, box = self.detail_single_patch(
                image = image,
                mask = mask,
                pos_prompt = pos_prompt, neg_prompt = neg_prompt,
                generator = generator,
                context_ratio = context_ratio,
                soft_blend_radis = soft_blend_radis,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                strength = strength,
                additional_diffusion_params = additional_diffusion_params,
            )
            detailed.paste(detailed_patch, (int(box[0]), int(box[1])), blend_mask)
        return detailed
    
    def detail_single_patch(self,
            image: Image.Image,
            mask: torch.Tensor,
            pos_prompt: str, neg_prompt: str,
            generator: None | torch.Generator = None,
            context_ratio: float = 0.25,
            num_inference_steps: int = 20,
            guidance_scale: float = 2.5,
            strength: float = 0.4,
            additional_diffusion_params: dict[str, Any] | None = None,
            soft_blend_radius: int = 3) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
        boxes = masks_to_boxes(mask.unsqueeze(0))
        mask_image = Image.fromarray(mask.numpy().astype(np.uint8)*255, mode="L")
        box = boxes[0].cpu().numpy()
        print(f"Detailer: Processing mask with bbox {tuple(box)}")
        growth_w = int((box[2] - box[0]) * context_ratio)
        growth_h = int((box[3] - box[1]) * context_ratio)
        box = box + [-growth_w, -growth_h, growth_w, growth_h]
        box = np.clip(box, 0, [image.size[0], image.size[1], image.size[0], image.size[1]])
        box = box.astype(np.uint32)
        print(f"Detailer: Expanded context bbox to {tuple(box)}")
        cropped = image.crop(tuple(box))
        cropped_mask = mask_image.crop(tuple(box))

        inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pipe(self._pipeline)
        assert isinstance(inpaint_pipe, StableDiffusionXLInpaintPipeline)
        upscaled = scale_largest_dimension_to(cropped)
        upscaled_mask = scale_largest_dimension_to(cropped_mask)
        print(f"Detailer: Upscaled diffusion dimension: {upscaled.size}")
        params: dict[str, Any] = dict(
            image = upscaled,
            mask_image = upscaled_mask,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            strength = strength,
        )
        prompt_embeds = get_sdxl_prompt_embeds(self._pipeline, pos_prompt, neg_prompt)
        params.update(prompt_embeds)
        if additional_diffusion_params:
            params.update(additional_diffusion_params)
        if not generator is None:
            params["generator"] = generator
        detailed_patch = inpaint_pipe(**params)[0][0] # pyright: ignore[reportIndexIssue]
        assert isinstance(detailed_patch, Image.Image) 
        detailed_final = detailed_patch.resize(cropped.size, Image.Resampling.LANCZOS)
        blend_mask = cropped_mask.filter(ImageFilter.GaussianBlur(radius=soft_blend_radis))
        print(f"Detailer: Finished mask with bbox {tuple(box)}")
        return (detailed_final, blend_mask, tuple(box))
