import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D,MaxPooling2D,Reshape,ZeroPadding2D,GlobalMaxPooling2D,GRU,Bidirectional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt

def fit_LSTM(trainX,trainY,save_file):    
    model = Sequential()
    model.add(LSTM(500, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))#lstm lag
    model.add(LSTM(500, activation='relu', return_sequences=True)) #lstm lag
    model.add(LSTM(500, activation='relu', return_sequences=False)) #lstm lag
    model.add(Dense(trainY.shape[1]))#NN lag
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(trainX, trainY, epochs=35, batch_size=16, validation_split=0.1, verbose=1)
    model.save(save_file)
    return model     

def fit_DNN(trainX,trainY,save_file):
    input_shape = (trainX.shape[1], trainX.shape[2])

    # Create a sequential model
    model = models.Sequential()

    # Add layers to the model
    model.add(layers.Flatten(input_shape=input_shape))  # Flatten the input
    model.add(layers.Dense(1000, activation='relu'))      # Dense layer with 128 units and ReLU activation                      # Dropout layer for regularization
    model.add(layers.Dense(1000, activation='relu'))       # Another Dense layer with 64 units and ReLU activation
    model.add(layers.Dense(1000, activation='relu'))       # Another Dense layer with 64 units and ReLU activation
    model.add(layers.Dense(trainY.shape[1], activation='relu'))    # Output layer with 10 units for classification (adjust as needed)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Display the model summary
    model.summary()
    model.fit(trainX, trainY, epochs=35, batch_size=16, validation_split=0.1, verbose=1)
    model.save(save_file)
    return model     

    

def split_dataframe_columns(df):
    """
    Splits a DataFrame into three based on column prefixes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: A tuple containing three DataFrames - (lmd_df, nwp_df, power_df).
    """
    lmd_columns = [col for col in df.columns if col.startswith("lmd")]
    nwp_columns = [col for col in df.columns if col.startswith("nwp")]
    power_columns = [col for col in df.columns if col.startswith("power")]

    lmd_df = df[lmd_columns]
    nwp_df = df[nwp_columns]
    power_df = df[power_columns]

    return lmd_df, nwp_df, power_df

def normalize_dataframes(*dfs):
    """
    Normalizes a list of DataFrames using Min-Max scaling.

    Parameters:
    - dfs (pd.DataFrame): Variable number of DataFrames to normalize.

    Returns:
    tuple: A tuple containing the normalized DataFrames.
    """
    scaler = MinMaxScaler()

    normalized_dfs = tuple(scaler.fit_transform(df) for df in dfs)

    return normalized_dfs


def remove_cols(data, cols_to_remove=['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_humidity','nwp_hmd_diffuseirrad','lmd_hmd_directirrad']):
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

def load_LSTM_data(station,cols_to_remove=None,n_future = 24 * 4,n_past = 4 * 4):
    station_name = os.path.splitext(station)[0]
    file_format = os.path.splitext(station)[1]
    file_path = station_name+'LSTM.pkl'
    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)
        print('and have been downloaded')
    else:
        print(f'The file {file_path} does not exist, so the dataset is being split up for the first time')
        if file_format == '.csv':
            data=fl.loadFile(station_name+'.csv',PKL=False)
        else:
            data = fl.loadPkl(station_name + '.pkl')

        data=remove_cols(data,cols_to_remove)
        
        lmd_data, nwp_data, power_data = split_dataframe_columns(data)

        # normalize the dataset
        if lmd_data.shape[1]>0:
            normalized_lmd, normalized_nwp, normalized_power = normalize_dataframes(lmd_data, nwp_data, power_data)
            normalized_lmd_train = normalized_lmd[:int(normalized_lmd.shape[0] * 0.8), :]
            normalized_lmd_test = normalized_lmd[int(normalized_lmd.shape[0] * 0.8):, :]
        else:
            normalized_nwp, normalized_power = normalize_dataframes(nwp_data, power_data)
        
        
        normalized_nwp_train = normalized_nwp[:int(normalized_nwp.shape[0] * 0.8), :]
        normalized_nwp_test = normalized_nwp[int(normalized_nwp.shape[0] * 0.8):, :]
        normalized_power_train = normalized_power[:int(normalized_power.shape[0] * 0.8), :]
        normalized_power_test = normalized_power[int(normalized_power.shape[0] * 0.8):, :]
        
        file_path = station_name+'LSTM.pkl'

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=4*4, n_future=24*4):
            X, Y = [], []
            if n_past>0:
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    past=lmd_data[i - n_past:i, :lmd_data.shape[1]]
                    future=nwp_data[i:i+n_future, :nwp_data.shape[1]]
                    combined_data = np.concatenate((past, future), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i+n_future])
            else:
                for i in range(0, len(nwp_data) - n_future+1 ):
                    future=nwp_data[i:i+n_future, :nwp_data.shape[1]]
                    X.append(future)
                    Y.append(power_data[i:i+n_future])
            return np.array(X), np.array(Y)
        if lmd_data.shape[1]>0:
            trainX, trainY = create_sequences(normalized_lmd_train,normalized_nwp_train,normalized_power_train, n_past, n_future)
            testX, testY = create_sequences(normalized_lmd_test,normalized_nwp_test,normalized_power_test, n_past, n_future)       
        else:
            trainX, trainY = create_sequences(nwp_data=normalized_nwp_train,power_data=normalized_power_train, n_past=n_past, n_future=n_future)
            testX, testY = create_sequences(nwp_data=normalized_nwp_test,power_data=normalized_power_test, n_past=n_past, n_future=n_future)

        
        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

        print(f'trainX shape == {trainX.shape}.')
        print(f'trainY shape == {trainY.shape}.')
        print(f'testX shape == {testX.shape}.')
        print(f'testY shape == {testY.shape}.')

    return trainX, trainY, testX, testY        
        
def get_only_day_data(datafile_path,trainY,testY,predictedData):
    """_summary_

    Args:
        datafile_path (string): file path of the file used for training and testing the model
        trainY (float): The power data used for training
        testY (float): The power data used for training
        predictedData (float): The predicted power based on the test datas features

    Returns:
        trainYDay: returns trainY but the nigth data cropped out
        testYDay: returns testY but the nigth data cropped out
        predictedDataDay: returns predictedData but the nigth data cropped out
    """
    if os.path.splitext(datafile_path)[1] == '.csv':
        data = fl.loadFile(datafile_path, PKL=False)
    else:
        data = fl.loadFile(datafile_path)
    data_temp = fl.loadPkl(os.path.splitext(datafile_path)[0] + '.pkl')
    tempDataIndex=0
    mask=[]
    testYDay=[]
    predictedDataDay=[]
    trainYDay=[]
    for i in range(0,len(data)):
        if tempDataIndex>len(data_temp)-1:
            break
        elif data.iloc[i,0] == data_temp.iloc[tempDataIndex,0]:
            tempDataIndex = tempDataIndex + 1
            if i<len(trainY):
                trainYDay =np.append(trainYDay,trainY[i,0,0])
            else:
                testYDay =np.append(testYDay,testY[i-len(trainY),0,0])
                predictedDataDay=np.append(predictedDataDay,predictedData[i-len(trainY),0])
    return trainYDay, testYDay, predictedDataDay
                

