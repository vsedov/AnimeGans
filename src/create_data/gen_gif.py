import argparse
import glob
import os

import imageio

from src.core import hc

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
    "--max_frames", type=int, default=1, help="Max number of frames in the gif"
)
args = parser.parse_args()

filenames = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
step = len(filenames) // args.max_frames if args.max_frames else 1
images = (imageio.imread(filename) for filename in filenames[::step])

imageio.mimsave(args.save_path, images, fps=8)
