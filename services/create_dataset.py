import numpy as np


def create_dataset(df, window_size=5):
    data = []
    labels = []
    
    for i in range(len(df) - window_size):
        # Get data for a time window
        window_data = df.iloc[i:i+window_size].values
        data.append(window_data)
        
        # If the closing price rises it is 1 and if it falls it is 0
        if df.iloc[i+window_size]["收盘价"] > df.iloc[i+window_size-1]["收盘价"]:
            labels.append(1)  
        else:
            labels.append(0)  
    
    return np.array(data), np.array(labels)