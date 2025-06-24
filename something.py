import pathlib
import os
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
        self.records = []
        dims = ['Kawaii', 'Warmth', 'Expressiveness']
        self.current_scores = {dim: 1 for dim in dims}

        # Set up figure and GridSpec with height_ratios to enlarge image area
        self.fig = plt.figure(figsize=(8, 10), facecolor="#7f7f7f")
        gs = self.fig.add_gridspec(
            nrows=5, ncols=5,
            height_ratios=[10, 1, 1, 1, 1],  # top row is 5x taller
            hspace=0, wspace=0
        )

        # Image spans top row
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Button grid: one row per dimension, 5 columns for ratings
        self.btns = {dim: [] for dim in dims}
        for i, dim in enumerate(dims):
            for j in range(5):
                ax = self.fig.add_subplot(gs[i+1, j])
                btn = Button(ax, str(j+1))
                btn.on_clicked(lambda _, val=j+1, d=dim: self.store_score(d, val))
                self.btns[dim].append(btn)
            # Label at the left of the row
            self.fig.text(0.01, 1 - (i+1)/5 - 0.05, f'{dim}:', va='center', fontsize=10)

        # Next button in bottom-right cell
        ax_next = self.fig.add_subplot(gs[4, 4])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.on_next)

        # Initial display
        self.update_display()
        plt.show()

    def store_score(self, dim, val):
        self.current_scores[dim] = val
        # Highlight selection
        for idx, btn in enumerate(self.btns[dim]):
            btn.color = 'lightgray'
            if idx + 1 == val:
                btn.color = 'skyblue'
        plt.draw()

    def update_display(self):
        self.ax_img.clear()
        cat, fname = self.images[self.idx]
        img = mpimg.imread(os.path.join(get_image_root(), cat, fname))
        self.ax_img.imshow(img)
        self.ax_img.axis('off')
        title = f'{self.idx+1}/{self.total} — {cat} | {fname.replace("blurred_",".").split(".")[0]}'
        self.ax_img.set_title(title)
        # reset buttons
        for dim, btn_list in self.btns.items():
            for _, btn in enumerate(btn_list):
                btn.color = 'lightgray'
            # highlight default (1)
            self.btns[dim][0].color = 'skyblue'
        plt.draw()

    def on_next(self, _):
        cat, fname = self.images[self.idx]
        pose = fname.replace('blurred_', '').rsplit('.', 1)[0]
        for dim, score in self.current_scores.items():
            self.records.append({
                'Category': cat,
                'Pose': pose,
                'Rating': dim,
                'RaterID': self.rater_id,
                'Score': float(score)
            })
        self.idx += 1
        if self.idx < self.total:
            # reset to defaults
            self.current_scores = {dim: 1 for dim in self.current_scores}
            self.update_display()
        else:
            plt.close(self.fig)

    def save(self, filename='results.csv'):
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f'All done! Saved to {filename}')


def main():
    # rater_id = input('Enter your email to start: ').strip()
    images = get_image_list()
    if not images:
        print('No images found in Pictures/.')
        return
    rater = ImageRater(images, "hi")# rater_id)
    rater.save()

if __name__ == '__main__':
    main()
