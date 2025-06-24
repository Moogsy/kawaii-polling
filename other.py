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
        # start with no scores
        self.current_scores = {dim: None for dim in dims}

        # 1) Pre-load all images into memory
        self.image_data = []
        for cat, fname in images:
            path = os.path.join(get_image_root(), cat, fname)
            self.image_data.append(mpimg.imread(path))

        # 2) Set up figure & GridSpec
        self.fig = plt.figure(figsize=(8, 10), facecolor="#7f7f7f")
        gs = self.fig.add_gridspec(
            nrows=5, ncols=5,
            height_ratios=[10, 1, 1, 1, 1],
            hspace=0, wspace=0
        )

        # Image axis spans top
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Button grid (no default highlights)
        self.btns = {dim: [] for dim in dims}
        for i, dim in enumerate(dims):
            for j in range(5):
                ax = self.fig.add_subplot(gs[i+1, j])
                btn = Button(ax, str(j+1))
                btn.color = 'lightgray'
                btn.on_clicked(lambda event, d=dim, v=j+1: self.store_score(d, v))
                self.btns[dim].append(btn)
            self.fig.text(0.01, 1 - (i+1)/5 - 0.05, f'{dim}:', va='center', fontsize=10)

        # Next button
        ax_next = self.fig.add_subplot(gs[4, 4])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.on_next)

        # Initial draw
        self.update_display()
        plt.show()

    def store_score(self, dim, val):
        self.current_scores[dim] = val
        # highlight only the clicked button
        for idx, btn in enumerate(self.btns[dim]):
            btn.color = 'lightgray'
            if idx + 1 == val:
                btn.color = 'skyblue'
        self.fig.canvas.draw_idle()

    def update_display(self):
        # metadata
        cat, fname = self.images[self.idx]
        pose = fname.replace('blurred_', '').rsplit('.', 1)[0]
        title = f'{self.idx+1}/{self.total} — {cat} | {pose}'

        img = self.image_data[self.idx]
        h, w = img.shape[:2]

        if not hasattr(self, 'img_artist'):
            # first time: create the AxesImage
            self.img_artist = self.ax_img.imshow(img)
            self.ax_img.axis('off')
        else:
            # just swap image data
            self.img_artist.set_data(img)

        # lock aspect to true image ratio
        self.ax_img.set_aspect(h / w, adjustable='box')
        self.ax_img.set_title(title)

        # reset all button highlights (none selected)
        for btn_list in self.btns.values():
            for btn in btn_list:
                btn.color = 'lightgray'

        self.fig.canvas.draw_idle()

    def on_next(self, event):
        # guard: require rating all dims
        if any(v is None for v in self.current_scores.values()):
            print("Please rate all dimensions before moving on.")
            return

        # record the scores
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

