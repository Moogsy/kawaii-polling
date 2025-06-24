from pathlib import Path

def get_image_root() -> Path:
    return Path(__file__).parent / "Pictures"

def get_image_list(): 
    root = get_image_root()

    images = []

    for img in root.glob("**/blurred_*.png"):
        category = img.parent
        images.append((category, img))

    return images
