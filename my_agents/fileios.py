# Written by Mingxuan Liu

import json
import os
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """
    lst: original list that to be split
    n  : number of partitions
    k  : chunk to get
    """
    chunks = split_list(lst, n)
    return chunks[k]


def ensure_folder_exists(folder_path):
    """Ensure that a folder exists. Create it if it does not."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def dump_json(filename, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")
    print("Successfully saved to {}".format(filename))


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def dump_txt(filename, in_data):
    if not filename.endswith('.txt'):
        filename += '.txt'

    with open(filename, 'w') as fbj:
        fbj.write(in_data)
    print("Successfully saved to {}".format(filename))


def load_txt(filename):
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as fbj:
        return fbj.read()
