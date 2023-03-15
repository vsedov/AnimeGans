import argparse
import os
import threading
from pathlib import Path

from PIL import Image
from imageio import v2 as imageio
from tqdm import tqdm

from src.core import hc


class ImageGridGif:
    """
    Class to create a gif from a directory of images with a grid layout.

    Attributes
    ----------
    img_dir : pathlib.Path
        Path to the directory containing the input PNG images.
    save_path : pathlib.Path
        Path to save the output gif.
    num_threads : int
        Number of threads to use for processing the images.

    Methods
    -------
    create_grid(img_path, crop_dir)
        Create a grid of 8x8 images from the input PNG file and save them to the specified directory.
    create_gif()
        Create a gif from the cropped images in the specified directory.
    """

    def __init__(self, args):
        """
        Initialize the ImageGridGif object.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments from the command line.
        """
        self.img_dir = Path(args.img_dir)
        self.save_path = Path(args.save_path)
        self.num_threads = args.num_threads

    def create_grid(self, img_path, crop_dir):
        """
        Create a grid of 8x8 images from the input PNG file and save them to the specified directory.

        Parameters
        ----------
        img_path : pathlib.Path
            Path to the input PNG file.
        crop_dir : pathlib.Path
            Path to the directory to save the cropped images.
        """
        img = Image.open(img_path)

        # Calculate the number of rows and columns for the grid
        img_width, img_height = img.size
        grid_cols = 8
        grid_rows = img_height // (img_width // grid_cols * 8)

        # Create a new directory to hold the cropped images
        crop_dir.mkdir(parents=True, exist_ok=True)

        # Crop out each 8x8 grid and save it to a new file
        for row in range(grid_rows):
            for col in range(grid_cols):
                x = col * img_width // grid_cols
                y = row * img_width // grid_cols * 8
                w = img_width // grid_cols
                h = img_width // grid_cols * 8
                crop_img = img.crop((x, y, x + w, y + h))
                crop_img.save(crop_dir / f"{row}_{col}.png")

    def create_gif(self):
        """
        Create a gif from the cropped images in the specified directory.
        """
        threads = []
        for img_path in self.img_dir.glob("*.png"):
            crop_dir = Path(os.path.splitext(img_path)[0])
            thread = threading.Thread(
                target=self.create_grid, args=(img_path, crop_dir)
            )
            thread.start()
            threads.append(thread)

        for thread, img_path in zip(threads, self.img_dir.glob("*.png")):
            thread.join()
            print(f"Processed {img_path}")
            tqdm.write(f"Processed {img_path}")

        # Create a gif from the cropped images
        with imageio.get_writer(self.save_path, mode="I", fps=8) as writer:
            for crop_subdir in self.img_dir.glob("*"):
                grid_filenames = sorted(crop_subdir.glob("*.png"))
                for filename in tqdm(grid_filenames):
                    writer.append_data(imageio.imread(str(filename)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a gif from a directory of images"
    )
    parser.add_argument(
        "--save_path",
        default=f"{hc.DIR}results/anime.gif",
        help="Path to save the gif",
    )
    parser.add_argument(
        "--img_dir",
        default=f"{hc.DIR}/results/",
        help="Directory containing the images",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1,
        help="Max number of frames in the gif",
    )
    parser.add_argument(
        "--step", type=int, default=10, help="Step between frames in the gif"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=30,
        help="Number of threads to use for processing the images",
    )
    args = parser.parse_args()

    image_grid_gif = ImageGridGif(args)
    image_grid_gif.create_gif()
