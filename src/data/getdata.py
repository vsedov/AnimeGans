import os
import zipfile

# Change this later
# from src.core import hc
# zip_path = f"{hc.DIR}/data/zip/"
# command = f"kaggle datasets download -d name/path -p {zip_path}"


class DSDownloader:
    def __init__(self, command, zip_path, unzip_dir, zip_name):
        self.command = command
        self.zip_location = zip_path
        self.unzip_dir = unzip_dir
        self.zip_name = zip_name

    def download_dir(self):
        if self.check_folder(self.unzip_dir):
            return
        self.check_zip()

        with zipfile.ZipFile(f"{self.zip_location}/{self.zip_name}", "r") as zip_ref:
            zip_ref.extractall(self.unzip_dir)

    def check_current_path_file(self, filename):
        return True if str(os.path.exists(filename)) else False

    def check_zip(self):
        if self.check_current_path_file(f"{self.zip_location}/{self.zip_name}"):
            os.system(self.command)

    def check_folder(self, folder):
        return True if os.path.isdir(folder) and os.listdir(folder) else False

    def force_replace_zip(self):
        if self.check_folder():
            for files in os.listdir(self.zip_location):
                os.remove(files)
        self.check_zip()

    def __call__(self):
        self.download_dir()
