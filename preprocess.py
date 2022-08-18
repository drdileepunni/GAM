
from tqdm import tqdm
from collections import defaultdict
import re
import pickle
import pandas as pd

from transform_functions import *

## SCRIPT DATA

select_cols = ['Hours', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Heart Rate',
       'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'SUBJECT_ID']

motors = ['Localizes Pain', 'Flex-withdraws', 'Obeys Commands',
       'No Response', 'Abnorm extensn', 'Abnorm flexion']
verbals = ['Confused', 'Inapprop words', 'ET/Trach', 'Oriented',
       'Incomp sounds']
eye = ['To speech', 'To pain', 'Spontaneously', 'No Response']

motors = [re.compile(suffix) for suffix in motors]
verbals = [re.compile(suffix) for suffix in verbals]
eye = [re.compile(suffix) for suffix in eye]

replace_dict = {
    'Glascow coma scale eye opening':eye, 
    'Glascow coma scale motor response':motors, 
    'Glascow coma scale verbal response':verbals
}

ffill_cols = ['Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']

mask_cols = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Heart Rate', 'Oxygen saturation', 
             'Respiratory rate', 'Systolic blood pressure']

## FUNCTIONS

def load_dataframes(tl_path, stays_path):
    '''
    Function takes timeline and stays dataframes from
    the data folder
    '''
    print('Loading dataframes')
    # Loading dataframes
    tl = pd.read_csv(tl_path)
    stays = pd.read_csv(stays_path)
    
    print(f"\nPP STEP 1: Selecting {len(select_cols)} columns out of {len(tl.columns)} columns..")
    tl = tl[select_cols]

    print(f'\nPP STEP 2: Replacing GCS..')
    for name, replace_ls in replace_dict.items():

        tl[name] = tl[name].apply(lambda x: remove_suffixes(x, replace_ls) if type(x)==str else x)
    
    print(f"\nPP STEP 3: Imputing {ffill_cols} using ffill strategy..")
    tl[ffill_cols] = tl[ffill_cols].ffill()
    
    tl["Mask"] = tl.loc[:, mask_cols].isna().sum(axis=1).apply(lambda x: 1 if x==0 else 0)
    empty_row_count = tl[tl['Mask']==0].shape[0]
    print(f"\nPP STEP 4: Mask feature added for rows that doesn't have at least one of {mask_cols} features..")
    print(f"{empty_row_count} rows masked out of {tl.shape[0]} rows..")
        
    return tl, stays

def generate_data_dict(timeline_df, stays, window_size=12):
    '''
    Main function that populates the data dictionary
    Structure of the data dictionary is as follows
    {
        subject_id: {
            'window': dataframe with features as column and number of windows as rows
            'target': this is the output variable after the last hour of timewindow (0-alive, 1-dead)
        }
    }
    '''
    # Creating group dictionary
    group_dict = {subj_id:timeline for subj_id, timeline in timeline_df.groupby('SUBJECT_ID')}

    data_dict = {} # Dictionary to store all data
    
    try:
        print('Transforming..')
        for subj_id, timeline in tqdm(group_dict.items()): # Processing individual timelines

            subj_dict = defaultdict(list) # Another dict thats provided as value to the data_dict

            mortality_status = get_mortality_status(subj_id, stays)

            # Ignoring invalid inputs

            if len(timeline)<window_size: 
                continue

            if mortality_status==None:
                continue

            # Breaking timelines to individual windows of a single 
            # timestep progression

            for i in range(len(timeline)):

                window_end = window_size+i

                if window_end<len(timeline):

                    window = timeline.iloc[i:window_end]
                    subj_dict['window'].append(reshape_window(window, window_size))

                    if (window_end==len(timeline)-1):

                        subj_dict['target'].append(mortality_status)

                    else:

                        subj_dict['target'].append(0)

            if subj_dict.get('window'):
                windows_df = pd.concat(subj_dict.get('window'))
                subj_dict['window'] = windows_df

                data_dict[subj_id] = subj_dict

        print('Storing data_dict in data folder..')
        with open('data/data_dict.pkl', 'wb') as file:
            pickle.dump(data_dict, file)
            
    except KeyboardInterrupt:
        
        print('Storing data_dict in data folder..')
        with open('data/data_dict_temp.pkl', 'wb') as file:
            pickle.dump(data_dict, file)
            
    return data_dict
