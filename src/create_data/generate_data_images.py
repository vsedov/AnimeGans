import os
import shutil
import tarfile


def open_tar_file(tar_file):
    """Open a tar file and return the file object."""
    return tarfile.open(tar_file, "r")


def main():
    tar_file = "data.tgz"
    if not os.path.isfile(tar_file):
        return FileNotFoundError(
            "File not found: {} Please Download this file".format(tar_file)
        )

    tar = open_tar_file(tar_file)

    tmp_dir = "./tmp"
    tar.extractall(tmp_dir)

    old_path = os.path.join(tmp_dir, "data", "images")
    new_path = os.path.join(tmp_dir, "data")
    os.rename(old_path, new_path)

    current_dir = os.path.abspath(".")
    test_folder_path = os.path.join(current_dir, "data")
    parent_dir = os.path.dirname(current_dir)
    shutil.move(test_folder_path, parent_dir)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
