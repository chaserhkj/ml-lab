import torch
import numpy as np
from torchvision.ops import masks_to_boxes # pyright: ignore[reportMissingTypeStubs]
from diffusers import StableDiffusionXLPipeline # pyright: ignore[reportPrivateImportUsage]
from diffusers import StableDiffusionXLInpaintPipeline # pyright: ignore[reportPrivateImportUsage]
from transformers import Sam3Processor, Sam3Model, logging
from PIL import Image, ImageFilter
from typing import Any

from . import (
    scale_largest_dimension_to,
    preview_image,
    shrink_bbox_to_div,
    shrink_bbox_to_ratio
)
from .conditioning import get_sdxl_prompt_embeds
# pyright: reportUnknownMemberType=none, reportAny=none, reportExplicitAny=none
# pyright: reportUnknownArgumentType=none, reportUnknownVariableType=none

logging.disable_progress_bar()

class SDXLDetailer(object):
    def __init__(self,
            pipeline: StableDiffusionXLPipeline,
            segment_prompt: str,
            segment_threshold: float = 0.5,
            mask_threshold: float = 0.5) -> None:
        self._pipeline: StableDiffusionXLPipeline = pipeline
        self._segment_prompt: str = segment_prompt
        self._segment_threshold: float = segment_threshold
        self._mask_threshold: float = mask_threshold
    def __call__(self,
                image: Image.Image,
                pos_prompt: str, neg_prompt: str,
                generator: None | torch.Generator = None,
                context_ratio: float = 0.25,
                merge_mask: bool = False,
                num_inference_steps: int = 20,
                guidance_scale: float = 2.5,
                strength: float = 0.4,
                additional_diffusion_params: dict[str, Any] | None = None,
                preview_mask: bool = False,
                crop_64_div: bool = True,
                force_patch_ratio: float | None = None,
                soft_blend_radius: int = 3) -> Image.Image:
        model = Sam3Model.from_pretrained("facebook/sam3")
        model = model.to("cuda", dtype=torch.float16) # pyright: ignore[reportArgumentType]
        processor = Sam3Processor.from_pretrained("facebook/sam3")

        inputs = processor(image, text=self._segment_prompt, return_tensors="pt").to("cuda")
        print(f"Detailer: Segmenting using prompt '{self._segment_prompt}'")
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=inputs["original_sizes"].tolist(), #pyright: ignore[reportAttributeAccessIssue]
            threshold=self._segment_threshold,
            mask_threshold=self._mask_threshold
        )[0]
        for i, s in enumerate(results["scores"]):
            bbox = results["boxes"][i].cpu().numpy().tolist()
            print(f"Detailer: Segmented {s.item()} @ {bbox}")
        masks = results["masks"]
        if len(masks) == 0:
            return image
        print(f"Detailer: Segmented {len(masks)} masks")
        if merge_mask:
            print("Detailer: Merging all masks")
            mask_combined = torch.any(masks, dim=0)
            masks = mask_combined.unsqueeze(0)

        masks = masks.cpu()
        detailed = image.copy()
        for mask in masks:
            detailed_patch, blend_mask, box = self.detail_single_patch(
                image = image,
                mask = mask,
                pos_prompt = pos_prompt, neg_prompt = neg_prompt,
                generator = generator,
                context_ratio = context_ratio,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                strength = strength,
                additional_diffusion_params = additional_diffusion_params,
                preview_mask = preview_mask,
                crop_64_div = crop_64_div,
                force_patch_ratio = force_patch_ratio,
                soft_blend_radius = soft_blend_radius,
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
            preview_mask: bool = False,
            crop_64_div: bool = True,
            force_patch_ratio: float | None = None,
            soft_blend_radius: int = 3) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
        boxes = masks_to_boxes(mask.unsqueeze(0))
        mask_image = Image.fromarray(mask.numpy().astype(np.uint8)*255, mode="L")
        if preview_mask:
            preview_image(mask_image)
        box = boxes[0].cpu().numpy()
        print(f"Detailer: Processing mask with bbox {box.tolist()}")
        growth_w = int((box[2] - box[0]) * context_ratio)
        growth_h = int((box[3] - box[1]) * context_ratio)
        box = box + [-growth_w, -growth_h, growth_w, growth_h]
        box = np.clip(box, 0, [image.size[0], image.size[1], image.size[0], image.size[1]])
        box = tuple(box.astype(np.uint32).tolist())
        print(f"Detailer: Expanded context bbox to {box}")
        if force_patch_ratio is not None:
            box = shrink_bbox_to_ratio(box, force_patch_ratio)
            print(f"Detailer: Force ratio {force_patch_ratio}, final bbox: {box}")
        cropped = image.crop(tuple(box))
        cropped_mask = mask_image.crop(tuple(box))

        inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pipe(self._pipeline)
        inpaint_pipe = inpaint_pipe.to(dtype=torch.float16)
        assert isinstance(inpaint_pipe, StableDiffusionXLInpaintPipeline)
        upscaled, upscale = scale_largest_dimension_to(cropped)
        upscaled_mask, _ = scale_largest_dimension_to(cropped_mask)
        print(f"Detailer: Upscaled dimension: {upscaled.size}")
        upscale_crop_box = (0, 0, upscaled.size[0], upscaled.size[1])
        if crop_64_div:
            upscale_crop_box = shrink_bbox_to_div(upscale_crop_box)
        print(f"Detailer: 64-div Crop: {upscale_crop_box}, size: ")
        patch = upscaled.crop(upscale_crop_box)
        patch_mask = upscaled_mask.crop(upscale_crop_box)
        print(f"Detailer: Diffusion dimensions: {patch.size}")

        downscale_size = (int(patch.size[0] / upscale), int(patch.size[1]/upscale))
        patch_origin = (
            box[0] + int(upscale_crop_box[0]/upscale),
            box[1] + int(upscale_crop_box[1]/upscale)
        )
        patch_bbox = (
            patch_origin[0], patch_origin[1],
            patch_origin[0] + downscale_size[0],
            patch_origin[1] + downscale_size[1]
        )
        print(f"Detailer: Final patch bbox: {patch_bbox}")

        params: dict[str, Any] = dict(
            image = patch,
            mask_image = patch_mask,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            strength = strength,
        )
        prompt_embeds = get_sdxl_prompt_embeds(self._pipeline, pos_prompt, neg_prompt)
        params.update(prompt_embeds)
        if additional_diffusion_params:
            params.update(additional_diffusion_params)
        if generator is not None:
            params["generator"] = generator
        detailed_patch = inpaint_pipe(**params)[0][0] # pyright: ignore[reportIndexIssue]
        assert isinstance(detailed_patch, Image.Image) 

        detailed_final = detailed_patch.resize(downscale_size, Image.Resampling.LANCZOS)
        final_mask = patch_mask.resize(downscale_size, Image.Resampling.LANCZOS)
        blend_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=soft_blend_radius))
        print(f"Detailer: Finished mask with bbox {box}")
        return (detailed_final, blend_mask, patch_bbox)
