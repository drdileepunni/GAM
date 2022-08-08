
import random
import pandas as pd
import numpy as np

split_ratio = 0.7
random_seed = 123

def train_test_split(data_dict, subset_prop=1, split_ratio=0.7, random_seed=123):

    subj_ids = list(data_dict.keys())
    random.Random(random_seed).shuffle(subj_ids)
    print(f"Random seed: {random_seed}")
    print(f"Split ratio: {split_ratio}")
    
    # Subsetting
    pre_subset_count = len(subj_ids)
    subset_index = round(len(subj_ids)*subset_prop)
    subj_ids = subj_ids[:subset_index]
    print(f"""Selected subset proportion: {subset_prop}
               Using {len(subj_ids)} of available {pre_subset_count} data points""")
    
    # Splitting
    split_index = round(len(subj_ids)*split_ratio)

    train_samples = subj_ids[:split_index]
    test_samples = subj_ids[split_index:]

    X_train, y_train = get_arrays(data_dict, train_samples)
    X_test, y_test = get_arrays(data_dict, test_samples)
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    final_split_ratio = train_size/(train_size+test_size)

    print(f"""Completed train test split.. Train size: {train_size} Test size: {test_size} 
        Final split ratio: {final_split_ratio}""")
    
    return X_train, X_test, y_train, y_test

def get_arrays(data_dict, subset):

    windows = [data_dict.get(sample).get('window') for sample in subset]
    targets = [data_dict.get(sample).get('target') for sample in subset]

    concat_df = pd.concat(windows)
    targets = [t for sublist in targets for t in sublist]

    X = concat_df.values
    y = np.array(targets)
    
    return X, y
