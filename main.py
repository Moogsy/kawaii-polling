import random

import json
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict
from matplotlib import image, pyplot as plt
from matplotlib import widgets
from matplotlib import image as mpimg


from samplers import sample_approx_2d
from image_rater import ImageRater


def get_image_root() -> Path:
    return Path(__file__).parent / "Pictures"


def get_image_list() -> list[tuple[Path, Path]]:
    root = get_image_root()
    images_per_category = defaultdict(list)
    for img in root.glob("**/blurred_*.png"):
        images_per_category[img.parent].append(img)

    # Quick sanity check. All values must have the same size for our
    # algorithm to work
    lengths = {len(v) for v in images_per_category.values()}
    assert len(lengths) == 1, f"Non uniform length found: {lengths}"

    pop_order = sample_approx_2d(len(images_per_category), lengths.pop())   
    keys: list[str] = list(images_per_category.keys())

    shuffled_images = []
    for category_index, img_index in pop_order:
        category_to_pop_from = keys[category_index]

        category_images = images_per_category[category_to_pop_from]

        image = category_images[img_index]
        shuffled_images.append((image.parent, image))

    return shuffled_images


if __name__ == "__main__":
    images = get_image_list()
    if not images:
        raise FileNotFoundError("No images found in Pictures/.")

    rater_name = input("Rater name: ")
    rater = ImageRater(images, rater_name)
    df = rater.save()

    try:
        path = "Ratings/"+ rater_name + "_ratings.csv"
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")
    except Exception:
        df.to_csv(sys.stdout, index=False)
    else:
        print("Done")

