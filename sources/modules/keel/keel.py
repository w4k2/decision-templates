import os
import io
import re

import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    attributes = re.findall(
        r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": np.int, "real": np.float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, np.object), types)]
    dtype = dict(zip(columns, types))

    data = pd.read_csv(io.StringIO(payload), names=columns, dtype=dtype)

    if not output:  # if it was not found
        output = columns[-1]

    target = data[output]
    data.drop(labels=output, axis=1, inplace=True)

    return data, target


def prepare_X_y(data, target):
    class_encoder = LabelEncoder()
    target = class_encoder.fit_transform(target.values.ravel())
    return data.values, target


def find_datasets(storage=STORAGE_DIR):
    for dirpath, _, filenames in os.walk(storage):
        dat_file = f"{os.path.basename(dirpath)}.dat"
        if dat_file in filenames:
            yield os.path.relpath(dirpath, storage)


def load_dataset(dataset_name, return_X_y=False, storage=STORAGE_DIR):
    data_file = os.path.join(storage, dataset_name, f"{dataset_name}.dat")
    data, target = parse_keel_dat(data_file)

    if return_X_y:
        return prepare_X_y(data, target)

    return Bunch(data=data, target=target, filename=data_file)
