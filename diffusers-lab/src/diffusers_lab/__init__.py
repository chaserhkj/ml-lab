from PIL import Image
from IPython.display import display
import numpy as np

BBox = tuple[int, int, int, int]

def preview_image(image:Image.Image, height: int=512) -> None:
    scale = height / image.size[1]
    width = int(image.size[0] * scale)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    _ = display(resized)

def scale_largest_dimension_to(image: Image.Image, length: int=1024) -> tuple[Image.Image, float]:
    if image.size[0] >= image.size[1]:
        scale = length / image.size[0]
    else:
        scale = length / image.size[1]
    new_w = int(image.size[0] * scale)
    new_h = int(image.size[1] * scale) 
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, scale

def grow_bbox_within_size(bbox: BBox, growth_ratio: float, size: tuple[int, int] | None = None) -> BBox:
    """Try to grow bbox by growth_ratio, if size is set, clip to size to prevent overflow"""
    growth_w = int((bbox[2] - bbox[0]) * growth_ratio)
    growth_h = int((bbox[3] - bbox[1]) * growth_ratio)
    new_bbox = np.array(bbox, dtype=np.uint32) + [-growth_w, -growth_h, growth_w, growth_h]
    if not size is None:
        new_bbox = np.clip(new_bbox, 0, [size[0], size[1], size[0], size[1]])
    new_bbox = tuple(new_bbox.tolist())
    return new_bbox

def shrink_bbox_to_div(bbox: BBox, div: int = 64) -> BBox:
    """Shrink bbox to ensure sides are divisible by div"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = width // div * div
    width_diff = width - new_width
    x1 += width_diff // 2
    x2 -= width_diff - width_diff // 2
    new_height = height // div * div
    height_diff = height - new_height
    y1 += height_diff // 2
    y2 -= height_diff - height_diff // 2
    return x1, y1, x2, y2

def shrink_bbox_to_ratio(bbox: BBox, target_ratio: float) -> BBox:
    """
    Shrinks a bounding box (x1, y1, x2, y2) to fit a target_ratio
    while keeping it centered on the original box's midpoint.
    """
    x1, y1, x2, y2 = bbox
    orig_w = x2 - x1
    orig_h = y2 - y1
    center_x = x1 + (orig_w // 2)
    center_y = y1 + (orig_h // 2)
    current_ratio = orig_w / orig_h
    if current_ratio > target_ratio:
        new_h = orig_h
        new_w = int(orig_h * target_ratio)
    else:
        new_w = orig_w
        new_h = int(orig_w / target_ratio)
    new_x1 = center_x - (new_w // 2)
    new_y1 = center_y - (new_h // 2)
    new_x2 = center_x + (new_w - new_w // 2)
    new_y2 = center_y + (new_h - new_h // 2)
    return (new_x1, new_y1, new_x2, new_y2)