import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as TF
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
strategy = TF.distribute.MirroredStrategy(
    cross_device_ops=TF.distribute.HierarchicalCopyAllReduce()
)

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

def loadPkl(file,path=None):
    if path==None:
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
    file_path=os.path.join(path,file)
    temp = open(file_path, 'rb')
    station = pickle.load(temp)
    temp.close()
    return station

def loadFile(file_name, path=None,PKL=True): 
    
    if path == None:
        print(f"Path of current program:\n", os.path.abspath(os.path.dirname(__file__)))
        datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/CSVFiles/'))
        # go one folder back to get to the base folder
        datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
        datafolder_path_csv =  datafolder_path+ "/CSVFiles/"

    else:
        datafolder_path_csv = path
    # display a warning if a pkl version exists
    if PKL:
        if (os.path.isfile(os.path.join(datafolder_path, file_name[:-4] + ".pkl"))):
            print("Warning: A pkl version of this file exists. It will be loaded instead of the csv file.")
            print("If you want to load the csv file, set PKL=False.")
            return loadPkl(file_name[:-4] + ".pkl",path)
    # check if folder exists if not then error
    if (os.path.isdir(datafolder_path_csv)):
        print(f"Path of dataset folder:\n", datafolder_path_csv)
    else:
        print("Data folder path does not exist")
        sys.exit()
    
    file_path = os.path.join(datafolder_path_csv, file_name)
    file_data=None
    # assign data in file
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path,header=0)
        if not(file_name == "metadata.csv"):
            file_data.index = pd.DatetimeIndex(file_data["date_time"])
    else:
        print("File name does not exist. Remember to include file type in file name")
        sys.exit()

    print("\n*** File succesfully loaded ***")
    print("\nFile preview:")
    print(file_data.head())
    return file_data

def remove_cols(data, cols_to_remove=['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_hmd_diffuseirrad','lmd_hmd_directirrad']):
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

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
        data_with_night = loadFile(station_file_path, PKL=False)
        data_without_night = loadFile(station_file_path, PKL=True)

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

def fit_LSTM(trainX, trainY, save_file, num_neurons=100, num_layers=1, epochs=10, batch_size=16, validation_split=0,learning_rate=0.001, optimizer=['RMSdrop','Adam']):
    with strategy.scope():
        model = TF.keras.Sequential()
        model.add(TF.keras.layers.LSTM(num_neurons, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True,kernel_initializer= TF.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=7))) # LSTM layer
        model.add(TF.keras.layers.LSTM(num_neurons, return_sequences=True, kernel_initializer= TF.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=8))) # LSTM layer
        model.add(TF.keras.layers.LSTM(num_neurons, return_sequences=True, kernel_initializer= TF.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=9))) # LSTM layer
        model.add(TF.keras.layers.LSTM(num_neurons, return_sequences=False,kernel_initializer= TF.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10))) # LSTM layer

        # Add an output layer with a dynamically determined number of neurons based on trainY.shape[1]
        #model.add(Dense(1, activation='linear', name='output'))
        model.add(TF.keras.layers.Dense(1, activation='linear', name='output'))
        if optimizer == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(lr=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(lr=learning_rate)
        # Compile the model with individual loss functions for each output
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Display the model summary
        model.summary()

        # Assuming you have training data (X_train, trainY)
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split,verbose=1)

        model.save(save_file)
    return model 

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
                    fit_LSTM(trainX,trainY,fit_save_file,num_neurons=num_neurons[iiii],num_layers=num_layers[iii],learning_rate=learning_rate[ii],optimizer = optimizer[i])

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
    result_row_list = []
    #testY =np.squeeze(testY, axis=(1, 2))
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
                    result_row = {"Model": prediction_save_file,'Actual':testY,'Predicted':predLSTM}
                    rows_list.append(row)
                    result_row_list.append(result_row)
    # Convert the list of rows to a DataFrame
    diffModelResult = pd.DataFrame(rows_list)

    diffModelResult.to_pickle(save_name+".pkl")
    
    diffModelData = pd.DataFrame(result_row_list)
    diffModelData.to_pickle("gridSearchData3.pkl")


#datafile
datafile_path = 'station01'

#indlæs data

trainX,trainY,testX,testY=load_with_time(datafile_path,96,96)

"""
learning_rate = [0.0001]
optimizer = ['Adam'] 
num_layers = [2]
num_neurons = [50]
"""

learning_rate = [0.01,0.001,0.0001]
optimizer = ['RMSprop','Adam'] 
num_layers = [3]
num_neurons = [50,100,200,400]


save_file = ""
parameter_grid_search_fit(save_file,trainX,trainY[:,95,0],learning_rate=learning_rate,optimizer=optimizer,num_layers=num_layers,num_neurons=num_neurons)

save_name="gridSearchResults3"

parameter_grid_search_prediction(save_file,save_name,testX,testY[:,95,0],learning_rate=learning_rate,optimizer=optimizer,num_layers=num_layers,num_neurons=num_neurons)


#RMSprop Adam



