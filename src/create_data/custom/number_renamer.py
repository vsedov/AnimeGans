import argparse
import os


def rename_files(path):
    files = [
        filename for filename in os.listdir(path) if filename.endswith(".png")
    ]
    for i, file in enumerate(files, 1):
        src = os.path.join(path, file)
        dst = os.path.join(path, f"{i}.png")
        os.rename(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename PNG files in a folder."
    )
    parser.add_argument(
        "path", type=str, help="Path to the folder containing PNG files."
    )
    args = parser.parse_args()
    rename_files(args.path)
