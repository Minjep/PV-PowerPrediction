import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D,MaxPooling2D,Reshape,ZeroPadding2D,GlobalMaxPooling2D,GRU,Bidirectional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

def fit_LSTM(trainX, trainY, save_file, num_neurons=100, num_layers=1, epochs=10, batch_size=16, validation_split=0,learning_rate=0.001,optimizer='adam'):
    model = Sequential()
    model.add(LSTM(num_neurons, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True)) # LSTM layer
    for i in range(num_layers - 1):
        model.add(LSTM(num_neurons, return_sequences=True)) # LSTM layer
    model.add(LSTM(num_neurons, return_sequences=False)) # LSTM layer

    # Add an output layer with a dynamically determined number of neurons based on trainY.shape[1]
    #model.add(Dense(1, activation='linear', name='output'))
    model.add(Dense(1, activation='linear', name='output'))

    # Compile the model with individual loss functions for each output
    model.compile(optimizer=optimizer, loss='mean_squared_error',learning_rate=learning_rate)

    # Display the model summary
    model.summary()

    # Assuming you have training data (X_train, trainY)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split,verbose=1)

    model.save(save_file)
    return model 

def fit_DNN(trainX,trainY,save_file):
    input_shape = (trainX.shape[1], trainX.shape[2])

    # Create a sequential model
    model = models.Sequential()

    # Add layers to the model
    model.add(layers.Flatten(input_shape=input_shape))  # Flatten the input
    model.add(layers.Dense(500, activation='relu'))      # Dense layer with 128 units and ReLU activation                      # Dropout layer for regularization
    model.add(layers.Dense(500, activation='relu'))       # Another Dense layer with 64 units and ReLU activation
    model.add(layers.Dense(500, activation='relu'))       # Another Dense layer with 64 units and ReLU activation
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

def remove_cols(data, cols_to_remove=['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_hmd_diffuseirrad','lmd_hmd_directirrad']):
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

def load_LSTM_data(station,cols_to_remove=None,n_future = 24 * 4,n_past = 1):
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

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=1, n_future=24*4):
            X, Y = [], []
            if n_past>0:
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    past=lmd_data[i - n_past:i, :lmd_data.shape[1]]
                    past_power = power_data[i-n_past:i]
                    past = np.hstack((past,past_power))
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

    return trainX, trainY, testX, testY

def load_LSTM_data_train_day_test_both(station,cols_to_remove=None,n_future = 24 * 4,n_past = 1):
    file_path = station+'_LSTM_data_train_day_test_both.pkl'
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
        data_day=fl.loadFile(station+'.pkl')
        data_night=fl.loadFile(station+'.csv',PKL=False)

        first_day_in_test=data_day.iloc[int(data_day.shape[0] * 0.8), 0]
        first_day_in_test = pd.to_datetime(first_day_in_test)

        data_day=remove_cols(data_day,cols_to_remove)
        data_night=remove_cols(data_night,cols_to_remove)
        data_night = data_night[data_day.columns]
        data_night = data_night.loc[data_night.index >= first_day_in_test]
        data_day = data_day.loc[data_day.index <= first_day_in_test]

        # Create a MinMaxScaler instance
        scaler = MinMaxScaler()

        # Fit and transform the DataFrame using MinMaxScaler
        scaler.fit(data_day)

        scaled_data_night = scaler.transform(data_night)
        scaled_data_day = scaler.transform(data_day)

        # Convert the scaled data back to a DataFrame
        data_day_scaled = pd.DataFrame(scaled_data_day, columns=data_day.columns, index=data_day.index)
        data_night_scaled = pd.DataFrame(scaled_data_night, columns=data_night.columns, index=data_night.index)

        lmd_data_day, nwp_data_day, power_data_day = split_dataframe_columns(data_day_scaled)
        lmd_data_night, nwp_data_night, power_data_night = split_dataframe_columns(data_night_scaled)

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=1, n_future=24*4):
            X, Y = [], []
            if n_past>0:
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    past=lmd_data.iloc[i - n_past:i, :lmd_data.shape[1]]
                    future=nwp_data.iloc[i:i+n_future, :nwp_data.shape[1]]
                    combined_data = np.concatenate((past, future), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i+n_future])
            else:
                for i in range(0, len(nwp_data) - n_future+1 ):
                    future=nwp_data.iloc[i:i+n_future, :nwp_data.shape[1]]
                    X.append(future)
                    Y.append(power_data.iloc[i:i+n_future])
            return np.array(X), np.array(Y)
        
        trainX, trainY = create_sequences(lmd_data_day, nwp_data_day, power_data_day, n_past, n_future)
        testX, testY = create_sequences(lmd_data_night, nwp_data_night, power_data_night, n_past, n_future)

        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

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
                trainYDay =np.append(trainYDay,trainY[i,:,0])
            else:
                testYDay.append(testY[i-len(trainY),:,0])
                predictedDataDay.append(predictedData[i-len(trainY),:])
    return trainYDay, testYDay, predictedDataDay

def load_LSTM_zero_padded(station_string, n_past=1, n_future=24*4):
    station_file_path=station_string+'.csv'
    file_path=station_string+'_0_padded.pkl'

    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)

        print('and have been downloaded')
    else:        
        # Load the DataFrames
        data_with_night = fl.loadFile(station_file_path, PKL=False)
        data_without_night = fl.loadFile(station_file_path, PKL=True)

        #remove colums 
        data_with_night_cols=remove_cols(data_with_night)

        #scale 
        scaler = MinMaxScaler()
        scaler.fit(data_with_night_cols)
        data_scaled=scaler.transform(data_with_night_cols)
        df_scaled_night = pd.DataFrame(data_scaled, columns=data_with_night_cols.columns, index=data_with_night_cols.index)

        # Extract the 'date_time' column values from both DataFrames
        date_time_without_night = data_without_night['date_time']

        # Create boolean masks based on whether 'date_time' values appear in both DataFrames
        mask_with_night = df_scaled_night.index.isin(date_time_without_night)

        # Set entries to a specific value where 'date_time' values do not appear in both DataFrames
        df_scaled_night.loc[~mask_with_night, :] = 0

        #split data
        lmd_data, nwp_data, power_data = split_dataframe_columns(df_scaled_night)
        #lmd_total,lmd_diffuse,lmd_temp,lmd_windspeed
        #nwp_global,nwp_direct,nwp,temp,nwp_windspeed

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=1, n_future=24*4):
                X, Y = [], []
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    past=lmd_data.iloc[i - n_past:i, :lmd_data.shape[1]]
                    future=nwp_data.iloc[i:i+n_future, :nwp_data.shape[1]]
                    past_power = power_data[i-n_past:i]
                    past = np.hstack((past,past_power))
                    # Find the number of missing columns in df1 compared to df2
                    """
                    num_missing_columns = len(future.columns) - len(past.columns)
                    # Add missing columns to df1 with zero values
                    for _ in range(num_missing_columns):
                        column_name = f'Zero_padded_{len(past.columns) + 1}'  # You can customize the column name as needed
                        past[column_name] = 0
                        """
                    combined_data = np.concatenate((past, future), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i+n_future])
                return np.array(X), np.array(Y)
        
        x_data,y_data=create_sequences(lmd_data,nwp_data,power_data, n_past=n_past, n_future=n_future)

        # Assuming x_data and y_data are your input and output data arrays
        total_samples = x_data.shape[0]
        split_index = int(0.8 * total_samples)

        # Training data
        trainX = x_data[:split_index]
        trainY = y_data[:split_index]

        # Testing data
        testX = x_data[split_index:]
        testY = y_data[split_index:]


        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

    return trainX,trainY,testX,testY

def parameter_grid_search_fit(save_file,trainX, trainY, learning_rate = [0.01,0.001,0.0001],optimizer = ['adam', 'RMSprop'],num_layers = [1,2,3],num_neurons = [100,200,400,800]):
    """_summary_
    This functions fits models for all combinations of 
    the parameters that it is giving in its arguments
    It will save all the models where the save_file is described

    Args:
        save_file (string): Where the fitted models should be save
                            Typically something like "grid_seach_models/"
        trainX (float): train data features
        trainY (float): train power actual values
        validation_split (list, optional): The different validations splits that should be tested. 
                                           Defaults to [0,0.1].
        batch_size (list, optional): The different batch sizes that should be tested. 
                                     Defaults to [4,8,16].
        num_layers (list, optional): The different amount of layers that should be tested.
                                     Defaults to [1,2,3].
        num_neurons (list, optional): The different number of neurons in each layer that should be tested. 
                                      Defaults to [100,200,400,800].
    """
    for i in range(len(optimizer)):
        for ii in range(len(learning_rate)):
            for iii in range(len(num_layers)):
                for iiii in range(len(num_neurons)):
                    fit_save_file = save_file + "optimizer"+str(optimizer[i])+"learnRate"+str(learning_rate[ii])+"lay"+str(num_layers[iii])+"neu"+str(num_neurons[iiii])+ ".keras"
                    fit_LSTM(trainX,trainY,fit_save_file,num_neurons=num_neurons[iiii],num_layers=num_layers[iii],learning_rate=learning_rate[ii],optimizer=optimizer[i])

def parameter_grid_search_prediction(model_path,save_name,testX, testY, learning_rate = [0.01,0.001,0.0001],optimizer = ['adam', 'RMSprop'],num_layers = [1,2,3],num_neurons = [100,200,400,800]):
    """_summary_
    When the different models is fitted in parameter_grid_search_fit
    This function can make predictions for the models and save them in
    a dataframe where the calculated mean squared error and R-squared
    is included for each parameter combination

    Args:
        model_path (string): Where the fitted models is saved
                             Typically something like "grid_seach_models/"
        save_name (string): What the data frame should be called and where it should be saved
                            E.g. "search_grid_dataframe"
        testX (float): test data features
        testY (float): test power actual values
        validation_split (list, optional): The different validations splits that are in the fitted models. 
                                           Defaults to [0,0.1].
        batch_size (list, optional): The different batch sizes that are in the fitted models. 
                                     Defaults to [4,8,16].
        num_layers (list, optional): The different amount of layers that are in the fitted models.
                                     Defaults to [1,2,3].
        num_neurons (list, optional): The different number of neurons in each layer that are in the fitted models. 
                                      Defaults to [100,200,400,800].
    """
    non_zero_indices = np.nonzero(testY)[0]
    testY = testY[non_zero_indices]
    
    rows_list = []
    testY =np.squeeze(testY, axis=(1, 2))
    for i in range(len(optimizer)):
        for ii in range(len(learning_rate)):
            for iii in range(len(num_layers)):
                for iiii in range(len(num_neurons)):
                    prediction_save_file = model_path + "optimizer" + str(optimizer[i]) + "learnRate" + str(learning_rate[ii]) + "lay" + str(num_layers[iii]) + "neu" + str(num_neurons[iiii]) + ".keras"
                    reconstructed_LSTM = keras.models.load_model(prediction_save_file)
                    predLSTM = reconstructed_LSTM.predict(testX)
                    predLSTM = np.squeeze(predLSTM, axis=1)
                    predLSTM = predLSTM[non_zero_indices]
                    mse1 = mse(testY, predLSTM)
                    R2 = r2_score(testY, predLSTM)
                    row = {'Optimizer': optimizer[i], 'Learning Rate': learning_rate[ii],'Num Layers': num_layers[iii], 'Num Neurons': num_neurons[iiii], 'MSE': mse1, 'R-squared': R2}
                    rows_list.append(row)
    # Convert the list of rows to a DataFrame
    diffModelResult = pd.DataFrame(rows_list)

    diffModelResult.to_pickle(save_name+".pkl")

def datetime_to_time_of_day(dt):
    # Extract hour, minute, and second components
    hours = dt.hour
    minutes = dt.minute
    seconds = dt.second

    # Convert to decimal representation
    time_of_day = hours.values + minutes.values / 60.0 + seconds.values / 3600.0

    #scale
    time_of_day=time_of_day/24

    # Use vectorized operation to calculate day of the year
    day_of_year = dt.dayofyear.values

    #scale
    day_of_year=day_of_year/365



    return time_of_day, day_of_year

def load_with_time(station_string, n_past=1, n_future=24*4):
    station_file_path=station_string+'.csv'
    file_path=station_string+'_0_padded.pkl'

    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)

        print('and have been downloaded')
    else:        
        # Load the DataFrames
        data_with_night = fl.loadFile(station_file_path, PKL=False)
        data_without_night = fl.loadFile(station_file_path, PKL=True)

        #remove colums 
        data_with_night_cols=remove_cols(data_with_night)

        #scale 
        scaler = MinMaxScaler()
        scaler.fit(data_with_night_cols)
        data_scaled=scaler.transform(data_with_night_cols)
        df_scaled_night = pd.DataFrame(data_scaled, columns=data_with_night_cols.columns, index=data_with_night_cols.index)

        # Extract the 'date_time' column values from both DataFrames
        date_time_without_night = data_without_night['date_time']

        # Create boolean masks based on whether 'date_time' values appear in both DataFrames
        mask_with_night = df_scaled_night.index.isin(date_time_without_night)

        # Set entries to a specific value where 'date_time' values do not appear in both DataFrames
        df_scaled_night.loc[~mask_with_night, :] = 0

        #split data
        lmd_data, nwp_data, power_data = split_dataframe_columns(df_scaled_night)
        #lmd_total,lmd_diffuse,lmd_temp,lmd_windspeed
        #nwp_global,nwp_direct,nwp,temp,nwp_windspeed

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=1, n_future=24*4):
                X, Y = [], []
                temp=(len(nwp_data) - n_future+1)
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    lmd=lmd_data.iloc[i - n_past:i, :lmd_data.shape[1]]
                    future=nwp_data.iloc[i:i+n_future, :nwp_data.shape[1]]
                    power=power_data.iloc[i - n_past:i]
                    past = pd.merge(lmd, power, left_index=True, right_index=True, how='left')
                    past_time,past_day=datetime_to_time_of_day(past.index)
                    past_df = pd.DataFrame({'time_of_day': past_time, 'day_of_year': past_day})
                    past['day_of_year'] = past_df['day_of_year'].values
                    past['time_of_day'] = past_df['time_of_day'].values


                    future_time,future_day=datetime_to_time_of_day(future.index)
                    future_df = pd.DataFrame({'time_of_day': future_time, 'day_of_year': future_day})
                    future['day_of_year'] = future_df['day_of_year'].values
                    future['time_of_day'] = future_df['time_of_day'].values
                        

                    # combined_data = pd.concat([past, future])
                    # combined_data = combined_data.fillna(0)
                    combined_data=np.concatenate((future.values,past.values))
                    X.append(combined_data)
                    Y.append(power_data[i:i+n_future].values)
                return np.array(X), np.array(Y)
        
        x_data,y_data=create_sequences(lmd_data,nwp_data,power_data, n_past=n_past, n_future=n_future)

        # Assuming x_data and y_data are your input and output data arrays
        total_samples = x_data.shape[0]
        split_index = int(0.8 * total_samples)

        # Training data
        trainX = x_data[:split_index]
        trainY = y_data[:split_index]

        # Testing data
        testX = x_data[split_index:]
        testY = y_data[split_index:]


        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

    return trainX,trainY,testX,testY

def load_data_daily_pred(station_string, n_past=1, n_future=24*4,time='00:00:00'):
    # Convert string to datetime object
    original_time = datetime.strptime(time, '%H:%M:%S')
    # Add 8 hours
    new_time = original_time + timedelta(hours=8)
    # Format the new time as a string
    time = new_time.strftime('%H:%M:%S')

    station_file_path=station_string+'.csv'
    file_path=station_string+'_0_padded.pkl'

    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)

        print('and have been downloaded')
    else:        
        # Load the DataFrames
        data_with_night = fl.loadFile(station_file_path, PKL=False)
        data_without_night = fl.loadFile(station_file_path, PKL=True)

        #remove colums 
        data_with_night_cols=remove_cols(data_with_night)

        #scale 
        scaler = MinMaxScaler()
        scaler.fit(data_with_night_cols)
        data_scaled=scaler.transform(data_with_night_cols)
        df_scaled_night = pd.DataFrame(data_scaled, columns=data_with_night_cols.columns, index=data_with_night_cols.index)

        # Extract the 'date_time' column values from both DataFrames
        date_time_without_night = data_without_night['date_time']

        # Create boolean masks based on whether 'date_time' values appear in both DataFrames
        mask_with_night = df_scaled_night.index.isin(date_time_without_night)

        # Set entries to a specific value where 'date_time' values do not appear in both DataFrames
        df_scaled_night.loc[~mask_with_night, :] = 0

        #split data
        lmd_data, nwp_data, power_data = split_dataframe_columns(df_scaled_night)
        #lmd_total,lmd_diffuse,lmd_temp,lmd_windspeed
        #nwp_global,nwp_direct,nwp,temp,nwp_windspeed

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=1, n_future=24*4,time='00:00:00'):
                occurrence_index=lmd_data.index.indexer_at_time(time)
                X, Y = [], []
                for index_value in occurrence_index:
                    if index_value >= n_past and len(power_data)-index_value>=n_future:
                        lmd=lmd_data.iloc[index_value - n_past+1:index_value+1,:]
                        future=nwp_data.iloc[index_value+1:index_value+n_future+1,:]
                        power=power_data.iloc[index_value - n_past+1:index_value+1,0]

                        past = pd.merge(lmd, power, left_index=True, right_index=True, how='left')
                        past_time,past_day=datetime_to_time_of_day(past.index)
                        past_df = pd.DataFrame({'time_of_day': past_time, 'day_of_year': past_day})
                        past['day_of_year'] = past_df['day_of_year'].values
                        past['time_of_day'] = past_df['time_of_day'].values

                        future_time,future_day=datetime_to_time_of_day(future.index)
                        future_df = pd.DataFrame({'time_of_day': future_time, 'day_of_year': future_day})
                        future['day_of_year'] = future_df['day_of_year'].values
                        future['time_of_day'] = future_df['time_of_day'].values
                            

                        # combined_data = pd.concat([past, future])
                        # combined_data = combined_data.fillna(0)
                        combined_data=np.concatenate((future.values,past.values))
                        X.append(combined_data)
                        Y.append(power_data[index_value+1:index_value+n_future+1].values)
                
                return np.array(X), np.array(Y)
        
        x_data,y_data=create_sequences(lmd_data,nwp_data,power_data, n_past=n_past, n_future=n_future,time=time)

        # Assuming x_data and y_data are your input and output data arrays
        total_samples = x_data.shape[0]
        split_index = int(0.8 * total_samples)

        # Training data
        trainX = x_data[:split_index]
        trainY = y_data[:split_index]

        # Testing data
        testX = x_data[split_index:]
        testY = y_data[split_index:]


        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

    return trainX,trainY,testX,testY