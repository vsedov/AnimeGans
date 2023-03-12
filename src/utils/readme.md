# Utils Folder

The utils folder contains helper functions and classes that are used throughout the project. In this folder, you will find the following files:

## Constants

This file contains dictionaries that store constant values used throughout the project. The dictionaries are hp and hc, which store configuration information for the project.
This is parsed through runtime

## core_helper

This file contains the CoreHelper class, which is used to parse data from the dictionaries in constants.py and store it as class attributes. This class can be used to access the configuration data at runtime.

## torch_utils.py

This file contains functions that help with labeling and defining elements from a dataset. These functions are used to perform operations such as data preprocessing, data augmentation, and data visualization.

## All other files

All other files are most likely MLOPS

# Usage

To use the helper functions and classes in the utils folder, simply import the
desired file and call the required function through `src.core [class](hp, hc)`

```python
from src.core import hc, hp
```

which are two core classes that contain the constants that are defined in
runtime.
