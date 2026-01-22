from PIL import Image
from IPython.display import display

def preview_image(image:Image.Image, height: int=512) -> None:
    scale = height / image.size[1]
    width = int(image.size[0] * scale)
    resized = image.resize((width, height), Image.LANCZOS)
    display(resized)

def scale_largest_dimension_to(image: Image.Image, length: int=1024) -> Image.Image:
    if image.size[0] >= image.size[1]:
        scale = length / image.size[0]
    else:
        scale = length / image.size[1]
    new_w = int(image.size[0] * scale)
    new_h = int(image.size[1] * scale) 
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized