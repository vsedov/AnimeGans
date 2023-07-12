import concurrent.futures
import glob
import os
import pickle
import re

import cv2
import i2v

DATA_DIR = "../../con"


def rename_files(dir_path):
    for file_name in os.listdir(dir_path):
        # skip directories and files that don't match the pattern
        if os.path.isdir(os.path.join(dir_path, file_name)):
            continue
        if not re.match(r"\d+_\d+\.(jpg|png)", file_name):
            continue

        # construct new filename and rename file
        new_file_name = re.sub(r"(\d+)_(\d+)\.(jpg|png)", r"\1\2.\3", file_name)
        os.rename(
            os.path.join(dir_path, file_name),
            os.path.join(dir_path, new_file_name),
        )
        print(f"Renamed {file_name} to {new_file_name}")


illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json"
)
rename_files(DATA_DIR)

files = glob.glob(f"{DATA_DIR}/*.jpg", recursive=True)
# files = glob.glob(f"{DATA_DIR}/*.png", recursive=True)

print(files)

tag_dict = [
    "orange hair",
    "white hair",
    "aqua hair",
    "gray hair",
    "green hair",
    "red hair",
    "purple hair",
    "pink hair",
    "blue hair",
    "black hair",
    "brown hair",
    "blonde hair",
    "gray eyes",
    "black eyes",
    "orange eyes",
    "pink eyes",
    "yellow eyes",
    "aqua eyes",
    "purple eyes",
    "green eyes",
    "brown eyes",
    "red eyes",
    "blue eyes",
]


def process_file(file):
    n_matches = re.findall(r"\d+", file)
    if len(n_matches) == 1:
        n = n_matches[0]
    elif len(n_matches) == 2:
        n = f"{n_matches[0]}_{n_matches[1]}"

    print(f"{n} - {file}")

    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (128, 128))
    feature = illust2vec.estimate_plausible_tags([img], threshold=0.25)
    f = feature[0]["general"]

    solo = False
    hair = False
    eye = False
    dictionary = {}
    for first, second in f:
        if first == "solo":
            solo = True
        if first in tag_dict:
            dictionary[first] = 1
            if "eyes" in first:
                eye = True
            if "hair" in first:
                hair = True

    # if the image is not solo, or the estimator hasn't found the color of eye or hair, skip the image because we need both
    # if solo is False or (eye is False and hair is False):
    #     print(file)
    #     return None

    return (n, dictionary)


def thread_function(files):
    d = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_file, files))
        for result in results:
            if result is not None:
                n, dictionary = result
                d[n] = dictionary

    return d


# Split the files into chunks to process them in batches
file_chunks = [files[i : i + 16] for i in range(0, len(files), 16)]
print(file_chunks)

results_dict = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    for i, result in enumerate(executor.map(thread_function, file_chunks)):
        results_dict |= result

        print(
            f"Processed {min((i+1)*16, len(files))} out of {len(files)} files Dict len: {len(results_dict)}"
        )

# Save the dictionary to disk
with open("../con.pickle", "wb") as handle:
    pickle.dump(results_dict, handle)
