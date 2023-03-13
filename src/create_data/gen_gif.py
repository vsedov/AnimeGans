import argparse
from pathlib import Path

from imageio import v2 as imageio
from tqdm import tqdm

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
parser.add_argument(
    "--step", type=int, default=5, help="Step between frames in the gif"
)
args = parser.parse_args()


with imageio.get_writer(args.save_path, mode="I", fps=8) as writer:
    filenames = sorted(Path(args.img_dir).glob("*.png"))
    filenames = [str(f) for f in filenames]
    print(len(filenames))

    for filename in tqdm(filenames[:: args.step]):
        writer.append_data(imageio.imread(filename))
