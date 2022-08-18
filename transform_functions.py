

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

    drop_cols = ['SUBJECT_ID', 'Hours']
    
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
get_mortality_status =  lambda subj_id, stays: stays[stays.SUBJECT_ID==subj_id].MORTALITY.values[0] \
                                if ((stays[stays.SUBJECT_ID==subj_id].MORTALITY.values==0) |\
                                    (stays[stays.SUBJECT_ID==subj_id].MORTALITY.values==1)) \
                                else None