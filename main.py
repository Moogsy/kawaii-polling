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


class ImageRater:
    SCALES = ("Kawaii", "Warmth", "Expressiveness")
    LIKERT = (
        "Strongly Disagree", "Disagree", "Neutral",
        "Agree", "Strongly Agree"
    )

    def __init__(self, images: list[tuple[Path, Path]], rater_id: str):
        self.images = images
        self.rater_id = rater_id
        self.idx = 0
        self.records = []
        # -1 means unanswered
        self.current_scores = {dim: -1 for dim in self.SCALES}

        self.fig = plt.figure(figsize=(8, 10), facecolor="#7f7f7F")
        gs = self.fig.add_gridspec(
            nrows=5, ncols=5,
            height_ratios=[10, 1, 1, 1, 1],
            hspace=0, wspace=0
        )

        # Image display axes
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Create Likert buttons
        self.buttons = {scale: [] for scale in self.SCALES}
        for si, scale in enumerate(self.SCALES):
            for li, label in enumerate(self.LIKERT):
                ax = self.fig.add_subplot(gs[si+1, li])
                btn = widgets.Button(ax, label)
                btn.on_clicked(lambda _, s=scale, lvl=li: self.store_score(s, lvl))
                # initialize patch color
                btn.ax.patch.set_facecolor("lightgray")
                self.buttons[scale].append(btn)

        # Next button (no color change needed)
        ax_next = self.fig.add_subplot(gs[4, 4])
        self.btn_next = widgets.Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)

        # Key press handler for numeric keys and Enter
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial display; blocks until window closed
        self.update_display()
        plt.show()

    def on_key(self, event):
        # Numeric keys 1-5 fill the next unanswered scale
        if event.key in [str(n) for n in range(1, 6)]:
            score = int(event.key)
            for scale in self.SCALES:
                if self.current_scores[scale] < 0:
                    self.store_score(scale, score - 1)
                    break
        # Enter key also triggers Next
        elif event.key == 'enter':
            self.on_next(event)

    def store_score(self, scale: str, agree_index: int):
        # Record the score (1-5)
        self.current_scores[scale] = agree_index + 1
        # Update each button's facecolor via its Axes patch
        for idx, btn in enumerate(self.buttons[scale]):
            color = "skyblue" if idx == agree_index else "lightgray"
            btn.ax.patch.set_facecolor(color)
            btn.color = color
        # Schedule redraw
        self.fig.canvas.draw_idle()

    def update_display(self):
        # Clear image axes and display the next image
        self.ax_img.clear()
        category, filepath = self.images[self.idx]
        img = mpimg.imread(get_image_root() / category / filepath)
        self.ax_img.imshow(img)
        self.ax_img.axis("off")
        # Show title
        title = f"{self.idx+1}/{len(self.images)} — {category.stem} | {filepath.stem}"
        self.ax_img.set_title(title)
        # Reset all Likert buttons to lightgray
        for blist in self.buttons.values():
            for btn in blist:
                btn.ax.patch.set_facecolor("lightgray")
                btn.color = "lightgray"
        # Schedule redraw
        self.fig.canvas.draw_idle()

    def on_next(self, _):
        # Only proceed if all scales have been scored
        if any(v < 0 for v in self.current_scores.values()):
            return
        # Append records for current image
        category, filepath = self.images[self.idx]
        pose = filepath.stem.replace("blurred_", "")
        for scale, score in self.current_scores.items():
            self.records.append({
                "Category": category.stem,
                "Model": pose,
                "Rating": scale,
                "RaterID": self.rater_id,
                "Score": score
            })
        # Move to next image or close
        self.idx += 1
        if self.idx < len(self.images):
            self.current_scores = {dim: -1 for dim in self.SCALES}
            self.update_display()
        else:
            plt.close(self.fig)

    def save(self):
        df = pd.DataFrame(self.records)
        try:
            df.to_csv(self.rater_id, index=False)
            print(f"Saved results to {self.rater_id}")
        except Exception:
            json.dump(self.records, sys.stdout)
        else:
            print("Done")


if __name__ == "__main__":
    images = get_image_list()
    if not images:
        raise FileNotFoundError("No images found in Pictures/.")
    rater = ImageRater(images, "ratings.csv")
    rater.save()

