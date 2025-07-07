from collections import defaultdict
from flask import Flask, render_template, request, redirect, session, url_for
from pathlib import Path
import pandas as pd
import samplers

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
SCALES = ["Kawaii", "Warmth", "Expressiveness"]
LIKERT = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
IMAGE_ROOT = Path(__file__).parent / "Pictures"
RATINGS_DIR = Path(__file__).parent / "Ratings"
RATINGS_DIR.mkdir(exist_ok=True)


def get_image_list():
    images_per_category = defaultdict(list)

    # Collect all blurred_*.png images under IMAGE_ROOT
    for img in IMAGE_ROOT.glob("**/blurred_*.png"):
        images_per_category[img.parent].append(img.relative_to(IMAGE_ROOT))

    # Map category → count
    counts = {str(cat.relative_to(IMAGE_ROOT)): len(imgs)
              for cat, imgs in images_per_category.items()}

    if not counts:
        raise ValueError(f"No images found under {IMAGE_ROOT}")
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        details = "; ".join(f"{cat}={count}" for cat, count in counts.items())
        raise ValueError(f"Non-uniform category sizes: {details}")

    common_size = unique_counts.pop()
    pop_order = samplers.sample_approx_2d(len(images_per_category), common_size)
    categories = list(images_per_category.keys())

    shuffled = []
    for cat_idx, img_idx in pop_order:
        cat = categories[cat_idx]
        img = images_per_category[cat][img_idx]
        shuffled.append(str(img))

    # Final integrity check
    all_images = {str(img) for imgs in images_per_category.values() for img in imgs}
    assert set(shuffled) == all_images, "Mismatch: missing or duplicate images"
    assert len(shuffled) == len(all_images), (
        f"Image count mismatch. Expected {len(all_images)} Got {len(shuffled)}"
    )

    return shuffled


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        rater = request.form.get('rater')
        if not rater:
            return render_template('home.html', error="Please enter your name.")
        session['rater'] = rater
        session['index'] = 0
        # Determine a unique output file for this rater
        base_file = RATINGS_DIR / f"web_{rater}_ratings.csv"
        if base_file.exists():
            i = 1
            while True:
                candidate = RATINGS_DIR / f"web_{rater}_ratings_{i}.csv"
                if not candidate.exists():
                    base_file = candidate
                    break
                i += 1
        session['outfile'] = str(base_file)

        # Prepare image list
        session['images'] = get_image_list()
        return redirect(url_for('rate'))
    return render_template('home.html')


@app.route('/rate', methods=['GET'])
def rate():
    idx = session.get('index', 0)
    images = session.get('images', [])
    if idx >= len(images):
        return redirect(url_for('thank_you'))
    image_path = images[idx]
    return render_template(
        'rate.html', image=image_path,
        scales=SCALES, likert=LIKERT,
        index=idx+1, total=len(images)
    )


@app.route('/submit', methods=['POST'])
def submit():
    rater = session.get('rater')
    idx = session.get('index')
    images = session.get('images', [])
    image_path = images[idx]
    category = Path(image_path).parent.name
    pose = Path(image_path).stem.replace("blurred_", "")

    records = []
    for scale in SCALES:
        score = request.form.get(scale)
        if score is None:
            return f"Missing score for {scale}", 400
        records.append({
            "Category": category,
            "Model": pose,
            "Rating": scale,
            "RaterID": rater,
            "Score": int(score)
        })

    df = pd.DataFrame(records)

    # Write to unique output file stored in session
    outfile = Path(session['outfile'])
    # If file didn't exist before this submission, write header
    write_header = not outfile.exists()
    df.to_csv(outfile, mode='a', header=write_header, index=False)

    session['index'] = idx + 1
    return redirect(url_for('rate'))


@app.route('/thank_you')
def thank_you():
    return "<h2>Thank you for completing the questionnaire!</h2>"


if __name__ == '__main__':
    app.run(debug=False)

