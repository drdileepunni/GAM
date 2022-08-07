
from tqdm import tqdm
from collections import defaultdict
import re
import pickle

## SCRIPT DATA

select_cols = ['Hours', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'SUBJECT_ID']

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

## FUNCTIONS

def remove_suffixes(s, patterns):
    '''
    Function to remove substrings from a string using 
    a list of regex patterns
    '''
    for pattern in patterns:
        s = pattern.sub("", s)
    return int(eval(s))

def reshape_window(window, window_size):
    '''
    Takes a timeline window and does the following transformations
    
    1. Removes unnecessary cols
    2. input df is of shape [window_size, (n_features+n_drop_cols+1)]
        reshapes this into [1, (window_size*n_features)]
    '''

    drop_cols = ['Temperature', 'SUBJECT_ID', 'Hours']
    
    window['timesteps'] = range(window_size)
    window = window.drop(drop_cols, axis=1)

    window_melt = window.melt(id_vars=['timesteps']).sort_values(['timesteps', 'variable'])
    
    # In this step we are joining two melted cols together and making it index 
    window_melt['feature'] = window_melt.apply(lambda x: x['variable'] + '_' + str(x['timesteps']), axis=1)
    window_melt = window_melt.drop(['variable', 'timesteps'], axis=1)[['feature', 'value']]

    # Finally transposing that the shape becomes [1, (window_size*n_features)] instead of 
    # [(window_size*n_features), 1]
    window_reshaped = window_melt.set_index('feature').transpose()
    
    return window_reshaped

# A function to get mortality status of the subject from stays dataframe
get_mortality_status =  lambda subj_id: stays[stays.SUBJECT_ID==subj_id].MORTALITY.values[0] \
                                if ((stays[stays.SUBJECT_ID==subj_id].MORTALITY.values==0) |\
                                    (stays[stays.SUBJECT_ID==subj_id].MORTALITY.values==1)) \
                                else None

def load_dataframes():
    '''
    Function takes timeline and stays dataframes from
    the data folder
    '''
    print('Loading dataframes')
    # Loading dataframes
    tl = pd.read_csv(tl_path)
    stays = pd.read_csv(stays_path)
    tl = tl[select_cols]


    print('Replacing GCS..')
    for name, replace_ls in replace_dict.items():

        tl[name] = tl[name].apply(lambda x: remove_suffixes(x, replace_ls))
        
    return tl, stays


def generate_data_dict(timeline_df, window_size=12):
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
    group_dict = {subj_id:timeline for subj_id, timeline in tl.groupby('SUBJECT_ID')}

    data_dict = {} # Dictionary to store all data

    print('Transforming..')
    for subj_id, timeline in tqdm(group_dict.items()): # Processing individual timelines

        subj_dict = defaultdict(list) # Another dict thats provided as value to the data_dict

        mortality_status = get_mortality_status(subj_id)

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
            
    return data_dict
