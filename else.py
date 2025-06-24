import os
from typing import Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

# Path to the Pictures folder
def get_image_root():
    return os.path.join(os.path.dirname(__file__), 'Pictures')

# Auto-generate list of (category, filename)
def get_image_list():
    root = get_image_root()
    images = []
    for category in sorted(os.listdir(root)):
        cat_path = os.path.join(root, category)
        if os.path.isdir(cat_path):
            for fname in sorted(os.listdir(cat_path)):
                if fname.lower().startswith('blurred_') and fname.lower().endswith('.png'):
                    images.append((category, fname))
    return images

class ImageRater:
    def __init__(self, images, rater_id):
        self.images = images
        self.total = len(images)
        self.rater_id = rater_id
        self.idx = 0
        self.records: list[dict[str, Any]] = []
        dims = ['Kawaii', 'Warmth', 'Expressiveness']
        # No default scores → start as None
        self.current_scores = {dim: None for dim in dims}

        # Pre-load all images into memory
        self.image_data = []
        for cat, fname in images:
            full_path = os.path.join(get_image_root(), cat, fname)
            self.image_data.append(mpimg.imread(full_path))

        # Set up figure and GridSpec
        self.fig = plt.figure(figsize=(8, 10), facecolor="#7f7f7f")
        gs = self.fig.add_gridspec(
            nrows=5, ncols=5,
            height_ratios=[20, 1, 1, 1, 1],
            hspace=0, wspace=0
        )

        # Image axis
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Buttons for each dimension
        self.btns = {dim: [] for dim in dims}
        for i, dim in enumerate(dims):
            for j in range(5):
                ax = self.fig.add_subplot(gs[i+1, j])
                btn = Button(ax, str(j+1))
                btn.on_clicked(lambda _, val=j+1, d=dim: self.store_score(d, val))
                btn.color = 'lightgray'  # all start unselected
                self.btns[dim].append(btn)
            # add row label
            self.fig.text(0.01, 1 - (i+1)/5 - 0.05, f'{dim}:', va='center', fontsize=10)

        # Next button
        ax_next = self.fig.add_subplot(gs[4, 4])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.on_next)

        # First display
        self.update_display()
        plt.show()

    def store_score(self, dim, val):
        # record choice
        self.current_scores[dim] = val
        # update highlights
        for idx, btn in enumerate(self.btns[dim]):
            btn.color = 'lightgray'
            if idx + 1 == val:
                btn.color = 'skyblue'
        self.fig.canvas.draw_idle()

    def update_display(self):
        cat, fname = self.images[self.idx]
        pose = fname.replace('blurred_', '').rsplit('.', 1)[0]
        title = f'{self.idx+1}/{self.total} — {cat} | {pose}'

        if not hasattr(self, 'img_artist'):
            # first time: create it
            self.img_artist = self.ax_img.imshow(self.image_data[self.idx])
            self.ax_img.axis('off')

            h, w = self.image_data[self.idx].shape[:2]
            self.ax_img.set_aspect(w / h, adjustable="box")
        else:
            # just swap the image data
            self.img_artist.set_data(self.image_data[self.idx])

        self.ax_img.set_title(title)

        # reset all button highlights (no default)
        for _dim, btn_list in self.btns.items():
            for btn in btn_list:
                btn.color = 'lightgray'

        self.fig.canvas.draw_idle()

    def on_next(self, _event):
        # ensure they rated every dimension
        if any(self.current_scores[dim] is None for dim in self.current_scores):
            print("Please rate all dimensions before moving on.")
            return

        # save the ratings
        cat, fname = self.images[self.idx]
        pose = fname.replace('blurred_', '').rsplit('.', 1)[0]
        for dim, score in self.current_scores.items():
            assert score is not None
            self.records.append({
                'Category': cat,
                'Pose': pose,
                'Rating': dim,
                'RaterID': self.rater_id,
                'Score': float(score)
            })

        self.idx += 1
        if self.idx < self.total:
            # reset for next image
            self.current_scores = {dim: None for dim in self.current_scores}
            self.update_display()
        else:
            plt.close(self.fig)

    def save(self, filename='results.csv'):
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f'All done! Saved to {filename}')

def main():
    images = get_image_list()
    if not images:
        print('No images found in Pictures/.')
        return
    rater_id = input('Enter your email to start: ').strip()
    rater = ImageRater(images, rater_id)
    rater.save()

if __name__ == '__main__':
    main()

