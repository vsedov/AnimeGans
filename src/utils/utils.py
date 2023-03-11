import os
from pathlib import Path


def project_root():
    return Path(__file__).parent.parent.parent


def current_path():
    return os.path.dirname(os.path.abspath(__file__))
