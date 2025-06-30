from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import widgets
import matplotlib.image as mpimg
import pandas as pd
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import widgets
import matplotlib.image as mpimg
import pandas as pd
import json
import sys

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import widgets
import matplotlib.image as mpimg
import pandas as pd
import json
import sys

def get_image_root() -> Path:
    return Path(__file__).parent / "Pictures"


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

        # Create figure with extra column for labels and definitions per scale
        self.fig = plt.figure(figsize=(8, 10), facecolor="#7f7f7F")
        gs = self.fig.add_gridspec(
            nrows=5, ncols=6,
            height_ratios=[10, 1, 1, 1, 1],
            width_ratios=[2, 1, 1, 1, 1, 1],
            hspace=0, wspace=0
        )

        # Image display axes (spanning all columns)
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Create Likert buttons with label+definition in first column
        self.buttons = {scale: [] for scale in self.SCALES}
        for si, scale in enumerate(self.SCALES):
            ax_label = self.fig.add_subplot(gs[si+1, 0])
            # Determine definition text
            if scale == "Warmth":
                title = "This pose is warm"
                definition = (
                    "Degree of perceived emotional warmth, trust, and friendliness."
                )
            elif scale == "Expressiveness":
                title = "This pose is expressive"
                definition = (
                    "Intensity and clarity of emotional expression"
                )
            else:  # Kawaii
                title = "This pose is kawaii"
                definition = "Left up to interpretation"
            # Draw bold label and definition separately
            ax_label.text(
                0.5, 0.65, title,
                ha="center", va="center",
                fontsize=10, fontweight="bold"
            )
            ax_label.text(
                0.5, 0.35, definition,
                ha="center", va="center",
                fontsize=9, wrap=True
            )
            ax_label.axis("off")

            # Likert buttons in columns 1-5
            for li, label in enumerate(self.LIKERT):
                ax = self.fig.add_subplot(gs[si+1, li+1])
                btn = widgets.Button(ax, label)
                btn.on_clicked(lambda _, s=scale, lvl=li: self.store_score(s, lvl))
                btn.ax.patch.set_facecolor("lightgray")
                self.buttons[scale].append(btn)

        # Next button in bottom-right cell
        ax_next = self.fig.add_subplot(gs[4, 5])
        self.btn_next = widgets.Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)

        # Key press handler for numeric keys and Enter
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial display; blocks until window closed
        self.update_display()
        plt.show()

    def on_key(self, event):
        if event.key in [str(n) for n in range(1, 6)]:
            score = int(event.key)
            for scale in self.SCALES:
                if self.current_scores[scale] < 0:
                    self.store_score(scale, score - 1)
                    break
        elif event.key == 'enter':
            self.on_next(event)

    def store_score(self, scale: str, agree_index: int):
        self.current_scores[scale] = agree_index + 1
        for idx, btn in enumerate(self.buttons[scale]):
            color = "skyblue" if idx == agree_index else "lightgray"
            btn.ax.patch.set_facecolor(color)
            btn.color = color
        self.fig.canvas.draw_idle()

    def update_display(self):
        self.ax_img.clear()
        category, filepath = self.images[self.idx]
        img = mpimg.imread(get_image_root() / category / filepath)
        self.ax_img.imshow(img)
        self.ax_img.axis("off")
        title = f"{self.idx+1}/{len(self.images)} — {category.stem}"
        self.ax_img.set_title(title)
        for blist in self.buttons.values():
            for btn in blist:
                btn.ax.patch.set_facecolor("lightgray")
                btn.color = "lightgray"
        self.fig.canvas.draw_idle()

    def on_next(self, _):
        if any(v < 0 for v in self.current_scores.values()):
            return
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
        self.idx += 1
        if self.idx < len(self.images):
            self.current_scores = {dim: -1 for dim in self.SCALES}
            self.update_display()
        else:
            plt.close(self.fig)

    def save(self):
        return pd.DataFrame(self.records)

