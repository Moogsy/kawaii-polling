import os
from PIL import Image
from pathlib import Path
import pandas as pd
import shutil

# Directory containing your PNG images
input_dir = Path("Pictures")
output_dir = Path("Pictures_Compressed")
output_dir.mkdir(exist_ok=True)

results = []

# Function to compress image until under 1MB
def compress_image(img_path, output_path, target_size=1_000_000):
    # initial quantize to reduce colors
    quality_levels = [256, 128, 64, 32, 16]
    for colors in quality_levels:
        with Image.open(img_path) as img:
            # Convert to palette-based and save
            img_quant = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
            img_quant.save(output_path, optimize=True)
        if output_path.stat().st_size <= target_size:
            break
    # if still too large, just leave the last version

# Process all PNGs
for img_path in input_dir.rglob("**/*.png"):
    rel_path = img_path.relative_to(input_dir)
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig_size = img_path.stat().st_size
    compress_image(img_path, out_path)
    new_size = out_path.stat().st_size

    res = {
        "Image": str(rel_path),
        "Original Size (KB)": round(orig_size/1024, 1),
        "Compressed Size (KB)": round(new_size/1024, 1)
    }
    results.append(res)
    
    print(res)


# Display results
df = pd.DataFrame(results)
import ace_tools as tools; tools.display_dataframe_to_user("Compression Results", df)

