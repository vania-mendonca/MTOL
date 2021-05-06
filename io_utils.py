import pickle
import pandas as pd


def print_to_txt(data, filepath):
    with filepath.open("w") as f:
        print(data, file=f)


def print_to_pickle(data, filepath):
    with filepath.open('wb') as f:
        pickle.dump(data, f)


def load_int_list_from_txt(filepath):
    read_lines = filepath.read_text(encoding="utf8")
    read_lines = read_lines.split("\n")
    return [int(line) for line in read_lines]


def load_list_from_pickle(filepath):
    data = []
    with filepath.open('rb') as f:
        data = pickle.load(f)
        
    return data


def load_dict_from_pickle(filepath):
    d = {}
    with filepath.open("rb") as f:
        d = pickle.load(f)
        
    return d


def load_dataframe_from_pickle(filepath):
    d = []
    with filepath.open("rb") as f:
        d = pickle.load(f)
    
    data = pd.DataFrame(d)
    return data