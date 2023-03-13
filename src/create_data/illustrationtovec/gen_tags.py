import concurrent.futures
import glob
import pickle
import re

import cv2
import i2v

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json"
)

files = glob.glob("../../data/*.png", recursive=True)
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
    n = re.findall(r"\d+", file)[-1]
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
    if solo == False or eye == False or hair == False:
        return None

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


file_chunks = [files[i : i + 16] for i in range(0, len(files), 16)]

# Use process-based parallelism to process each chunk
results_dict = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    for i, result in enumerate(executor.map(thread_function, file_chunks)):
        results_dict.update(result)

        print(
            f"Processed {min((i+1)*16, len(files))} out of {len(files)} files"
        )

with open("../features.pickle", "wb") as handle:
    pickle.dump(results_dict, handle)
