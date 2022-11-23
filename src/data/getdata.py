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
        """
        Download Dir:

        Function will download the path and everything related to the package
        Function will call the following

        Check Folder: data.getdata.DSDownloader.check_folder
        Check Zip: data.getdata.DSDownloader.check_zip
        and then will extract any files within the zip file it self.

        """
        if self.check_folder(self.unzip_dir):
            return
        self.check_zip()

        with zipfile.ZipFile(f"{self.zip_location}/{self.zip_name}", "r") as zip_ref:
            zip_ref.extractall(self.unzip_dir)

    def check_current_path_file(self, filename):
        """Check current file path
        filename : Filename to check
            Mainly used to check if the zip file exists

        Returns
        -------
        Boolean
            True if it does , else False

        """
        return True if str(os.path.exists(filename)) else False

    def check_zip(self):
        """
        Check Zip
        Checks if the file  path of self.zip_location/self.zip_name exists
        Function calls :data/getdata.py::DSDownloader::check_current_path_file
        """

        if self.check_current_path_file(f"{self.zip_location}/{self.zip_name}"):
            os.system(self.command)

    def check_folder(self, folder):
        """Check Folder
        folder : Folder: to check if it exists
        Returns
        -------
        Boolean
            if the folder exists return True else False
        """
        return True if os.path.isdir(folder) and os.listdir(folder) else False

    def force_replace_zip(self):
        """
        Force Replace the zip file
        removes everything in the folders
        Calls : data.getdata.DSDownloader.check_zip
        """
        if self.check_folder():
            for files in os.listdir(self.zip_location):
                os.remove(files)
        self.check_zip()

    def __call__(self):
        """
        Simple call function: Will call :
        Download Dir : data.getdata.DSDownloader.download_dir
        """
        self.download_dir()
