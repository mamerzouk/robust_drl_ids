import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def format_CICIoV2024(raw_data_path: str, split=0.7):

    # import the dataset into pandas DataFrames
    data_benign = pd.read_csv(raw_data_path+'/binary/binary_benign.csv')
    data_benign['labels'] = 'normal'

    data_attack = pd.read_csv(raw_data_path+'/binary/binary_DoS.csv')
    data_attack = pd.concat([data_attack,
                             pd.read_csv(raw_data_path+'/binary/binary_spoofing-GAS.csv'),
                             pd.read_csv(raw_data_path+'/binary/binary_spoofing-RPM.csv'),
                             pd.read_csv(raw_data_path+'/binary/binary_spoofing-SPEED.csv'),
                             pd.read_csv(raw_data_path+'/binary/binary_spoofing-STEERING_WHEEL.csv')])
    data_attack['labels'] = data_attack['category'].apply(lambda x: x.lower())

    data = pd.concat([data_benign, data_attack])
    data.drop(['label','category', 'specific_class'], inplace=True, axis=1)
    data = data.sample(frac=1)
    size_train = int(data.shape[0]*split)

    # split training and testing sets
    formated_train = data[:size_train]    
    formated_test = data[size_train:]

    means = formated_train.iloc[:, :-1].mean()
    stds = formated_train.iloc[:, :-1].std()

    return formated_train, formated_test, means, stds