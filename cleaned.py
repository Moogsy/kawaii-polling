import json
import sys

import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import widgets
from matplotlib import image as mpimg

def get_image_root() -> Path:
    return Path(__file__).parent / "Pictures"

def get_image_list() -> list[tuple[Path, Path]]: 
    root = get_image_root()

    images = []

    for img in root.glob("**/blurred_*.png"):
        category = img.parent
        images.append((category, img))

    return images

class ImageRater:

    SCALES = ("Kawaii", "Warmth", "Expressiveness")

    LIKERT = (
        "Strongly Disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly Agree"
    )

    def __init__(self, images: list[tuple[Path, Path]], rater_id: str):
        self.images = images
        self.images_count = len(images)

        self.rater_id = rater_id
        self.idx = 0
        self.records = []

        self.current_scores = {dim: 3 for dim in self.SCALES}

        self.fig = plt.figure(
            figsize=(8, 10),
            facecolor="#7f7f7F"
        )

        gridspec = self.fig.add_gridspec(
            nrows=5, ncols=5,
            height_ratios=[10, 1, 1, 1, 1], # Top row is 10x bigger,
            hspace=0, wspace=0
        )

        # The image is the whole top row
        self.ax_img = self.fig.add_subplot(gridspec[0, :]) 

        # Generate buttons
        self.buttons = {scale: [] for scale in self.SCALES}
        for scale_index, scale in enumerate(self.SCALES):
            for agree_index, agreement_level in enumerate(self.LIKERT):
                ax = self.fig.add_subplot(gridspec[scale_index + 1, agree_index])
                button = widgets.Button(ax, agreement_level)

                # Quick and dirty hack, matplotlib only passes
                # an event to this, so we store the values as the
                # function's default arguments
                button.on_clicked(
                    lambda _, s=scale, lvl=agree_index: self.store_score(s, lvl)
                )

                self.buttons[scale].append(button)

        # Finally, the next button
        ax_next = self.fig.add_subplot(gridspec[4, 4])
        self.btn_next = widgets.Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)

        self.update_display()

        plt.show()

    def store_score(self, scale: str, agree_index: int):
        self.current_scores[scale] = agree_index + 1

        for idx, btn in enumerate(self.buttons[scale]):
            if idx == agree_index:
                btn.color = "skyblue"
            else:
                btn.color = "lightgray"

        plt.draw()

    def update_display(self):
        self.ax_img.clear()
        cat, fname = self.images[self.idx]
        img = mpimg.imread(get_image_root() / cat / fname)
        self.ax_img.imshow(img)
        self.ax_img.axis("off")

        title = f'{self.idx+1}/{len(self.images)} — {cat.stem} | {fname.stem}'
        self.ax_img.set_title(title)

        for buttons_list in self.buttons.values():
            for btn in buttons_list:
                btn.color = "lightgray"

        plt.draw()


    def on_next(self, _):
        category, fname = self.images[self.idx]
        pose = fname.stem.replace("blurred_", "")

        for scale, score in self.current_scores.items():
            self.records.append({
                "Category": category.stem,
                "Model": pose,
                "Rating": scale,
                "RaterID": self.rater_id,
                "Score": score
            })

        self.idx += 1

        if self.idx < len(self.images):
            self.current_scores = {dim: 3 for dim in self.current_scores}
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


def main():
    images = get_image_list()
    if not images:
        raise FileNotFoundError("No images found in Pictures/.")

    rater = ImageRater(images, "hi.csv")
    rater.save()


if __name__ == "__main__":
    main()
