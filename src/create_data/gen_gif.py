import argparse
import os
import threading
from pathlib import Path

from PIL import Image
from imageio import v2 as imageio
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from src.core import hc

parser = argparse.ArgumentParser(
    description="Create a gif from a directory of images"
)
parser.add_argument(
    "--save_path",
    default=f"{hc.DIR}results/anime",
    help="Path to save the gif",
)
parser.add_argument(
    "--img_dir",
    default=f"{hc.DIR}/results/",
    help="Directory containing the images",
)
parser.add_argument(
    "--format",
    choices=["mp4", "gif"],
    default="mp4",
    help="Output format (default: mp4)",
)
args = parser.parse_args()
save_path = (
    args.save_path + ".mp4" if args.format == "mp4" else args.save_path + ".gif"
)

# Define a function to create an image file from the 8x8 grid images
def create_grid(img_path, output_dir):
    img = Image.open(img_path)

    # Calculate the number of rows and columns for the grid
    img_width, img_height = img.size
    grid_cols = 8
    grid_rows = img_height // (img_width // grid_cols * 8)

    # Create a new image to hold the 8x8 grid
    grid_img = Image.new(mode="RGB", size=(img_width, img_width))

    # Crop out each 8x8 grid and paste it onto the grid image
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * img_width // grid_cols
            y = row * img_width // grid_cols * 8
            w = img_width // grid_cols
            h = img_width // grid_cols * 8
            crop_img = img.crop((x, y, x + w, y + h))
            grid_img.paste(crop_img, (x, y))

    # Save the grid image to a file
    output_path = os.path.join(
        output_dir,
        "merged_{}.png".format(os.path.splitext(os.path.basename(img_path))[0]),
    )
    grid_img.save(output_path)


# Create a thread for each image file and crop out each 8x8 grid concurrently
threads = []
output_dir = os.path.join(args.img_dir, "merged")
os.makedirs(output_dir, exist_ok=True)

for img_path in Path(args.img_dir).glob("*.png"):
    thread = threading.Thread(target=create_grid, args=(img_path, output_dir))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Create a gif from the merged grid images
merged_files = sorted(Path(output_dir).glob("*.png"))

if args.format == "mp4":
    with tqdm(total=len(merged_files), desc="Creating mp4") as pbar:
        clip = ImageSequenceClip([str(f) for f in merged_files], fps=8)
        clip.write_videofile(
            save_path, fps=8, codec="libx264", bitrate="30000k"
        )
        pbar.update(1)
else:
    with tqdm(total=len(merged_files), desc="Creating gif") as pbar:
        with imageio.get_writer(save_path, mode="I", fps=8) as writer:
            for filename in merged_files:
                frame = imageio.imread(str(filename))
                writer.append_data(frame)
                pbar.update(1)
                frame = (
                    None  # set the frame to None to release the image memory
                )
