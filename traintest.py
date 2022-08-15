import random
import pandas as pd
import numpy as np
from tqdm import tqdm

split_ratio = 0.7
random_seed = 123

def train_test_split(data_dict, subset_prop=1, subset_level='admission', split_ratio=0.7, random_seed=random_seed):

    subj_ids = list(data_dict.keys())
    random.Random(random_seed).shuffle(subj_ids)
    print(f"Random seed: {random_seed}")
    print(f"Split ratio: {split_ratio}")
    
    # Subsetting at admission level
    if subset_level=='admission':
        pre_subset_count = len(subj_ids)
        subset_index = round(len(subj_ids)*subset_prop)
        subj_ids = subj_ids[:subset_index]
        print(f"""Selected subset proportion: {subset_prop}
                   Using {len(subj_ids)} of available {pre_subset_count} data points""")
    
    # Splitting
    split_index = round(len(subj_ids)*split_ratio)

    train_samples = subj_ids[:split_index]
    test_samples = subj_ids[split_index:]
    
    if subset_level=='window':
        X_train, y_train = get_arrays(data_dict, train_samples, sample_prop=subset_prop)
        X_test, y_test = get_arrays(data_dict, test_samples, sample_prop=subset_prop)
    else:
        X_train, y_train = get_arrays(data_dict, train_samples)
        X_test, y_test = get_arrays(data_dict, test_samples)
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    final_split_ratio = train_size/(train_size+test_size)

    print(f"""Completed train test split.. Train size: {train_size} Test size: {test_size} 
        Final split ratio: {final_split_ratio}""")
    
    return X_train, X_test, y_train, y_test

def get_arrays(data_dict, subset, subset_prop=1, sample_prop=None):
    
    if sample_prop:
        
        windows = []; targets = []
        for sample in tqdm(subset):
            
            window = data_dict.get(sample).get('window')
            target = data_dict.get(sample).get('target')
            
            window, target = subsample_window(window, target, sample_prop)
            
            windows.append(window)
            targets.append(target)
    else:
        
        windows = [data_dict.get(sample).get('window') for sample in subset]
        targets = [data_dict.get(sample).get('target') for sample in subset]

    concat_df = pd.concat(windows)
    targets = [t for sublist in targets for t in sublist]

    X = concat_df.values
    y = np.array(targets)
    
    return X, y

def subsample_window(window, target, sample_prop=0.2, seed=random_seed):
    '''
    Function takes in a window df and target array. Subsets a proportion 
    of both df and the array using random indices. In this process, the 
    last row of the df and last element of the array are preserved as these
    are the rows with positive samples.
    
    window: pandas df
    target: list
    sample_prop: float, percentage of rows/list elements to be randomly sampled
    '''
    random.seed(seed)
    
    n_window_rows = window.shape[0]
    sample_size = round(n_window_rows*sample_prop)

    # Getting row indices to subsample
    sample_row_indices = [random.randrange(1, (n_window_rows-1), 1) for i in range(sample_size)]

    # Subsampling window using indices
    window_wo_last_row = window.iloc[:-1,:]
    last_row = window.iloc[[-1],:]
    sample_df_wo_last_row = window_wo_last_row.iloc[sample_row_indices, :]
    window_subset = pd.concat([sample_df_wo_last_row, last_row])

    # Subsampling targets using indices
    target_wo_last = [target[i] for i in sample_row_indices]
    target_subset = target_wo_last + [target[-1]]
    
    assert len(window_subset) == len(target_subset)

    return window_subset, target_subset