import os,sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D,MaxPooling2D,Reshape,ZeroPadding2D,GlobalMaxPooling2D,GRU,Bidirectional
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import LSTM as LS
from sklearn.metrics import r2_score, mean_squared_error
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin
from keras.wrappers.scikit_learn import KerasRegressor

#trainX, trainY, testX, testY = LS.load_LSTM_zero_padded('station02', 24 * 4, 24 * 4)
trainX, trainY, testX, testY = LS.load_with_time('station01', 24 * 4, 24 * 4)

def load_CNN_data(station, cols_to_remove=None, n_future=24 * 4, n_past=1):
    station_name = os.path.splitext(station)[0]
    file_format = os.path.splitext(station)[1]
    file_path = station_name + 'LSTM.pkl'
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
            data = fl.loadFile(station_name + '.csv', PKL=False)
        else:
            data = fl.loadPkl(station_name + '.pkl')

        data = LS.remove_cols(data, cols_to_remove)

        lmd_data, nwp_data, power_data = LS.split_dataframe_columns(data)

        # normalize the dataset
        if lmd_data.shape[1] > 0:
            normalized_lmd, normalized_nwp, normalized_power = LS.normalize_dataframes(lmd_data, nwp_data, power_data)
            normalized_lmd_train = normalized_lmd[:int(normalized_lmd.shape[0] * 0.8), :]
            normalized_lmd_test = normalized_lmd[int(normalized_lmd.shape[0] * 0.8):, :]
        else:
            normalized_nwp, normalized_power = LS.normalize_dataframes(nwp_data, power_data)

        normalized_nwp_train = normalized_nwp[:int(normalized_nwp.shape[0] * 0.8), :]
        normalized_nwp_test = normalized_nwp[int(normalized_nwp.shape[0] * 0.8):, :]
        normalized_power_train = normalized_power[:int(normalized_power.shape[0] * 0.8), :]
        normalized_power_test = normalized_power[int(normalized_power.shape[0] * 0.8):, :]

        file_path = station_name + 'LSTM.pkl'

        def create_sequences(lmd_data=None, nwp_data=None, power_data=None, n_past=1, n_future=24 * 4):
            X, Y = [], []
            if n_past > 0:
                for i in range(n_past, len(nwp_data) - n_future + 1):
                    past = lmd_data[i - n_past:i, :lmd_data.shape[1]]
                    future = nwp_data[i:i + n_future, :nwp_data.shape[1]]
                    combined_data = np.concatenate((past.flatten(), future.flatten()), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i + n_future])
            else:
                for i in range(0, len(nwp_data) - n_future + 1):
                    future = nwp_data[i:i + n_future, :nwp_data.shape[1]]
                    X.append(future)
                    Y.append(power_data[i:i + n_future])
            return np.array(X), np.array(Y)

        if lmd_data.shape[1] > 0:
            trainX, trainY = create_sequences(normalized_lmd_train, normalized_nwp_train, normalized_power_train,
                                              n_past, n_future)
            testX, testY = create_sequences(normalized_lmd_test, normalized_nwp_test, normalized_power_test, n_past,
                                            n_future)
        else:
            trainX, trainY = create_sequences(nwp_data=normalized_nwp_train, power_data=normalized_power_train,
                                              n_past=n_past, n_future=n_future)
            testX, testY = create_sequences(nwp_data=normalized_nwp_test, power_data=normalized_power_test,
                                            n_past=n_past, n_future=n_future)

        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

    return trainX, trainY, testX, testY


def fit_CNN_tune(hp):
    # input must be of shape [batch_size, time_steps, input_dimension]
    #trainX_reshaped = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    n_timesteps = trainX.shape[1]  # 192
    n_features = trainX.shape[2]  # 7
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    # filters is the nr. of neurons
    # kernel size here is 1D because of the type of CNN, which is Conv1D
    # model.add(keras.layers.Conv1D(hp.Int('conv1_units', min_value=300, max_value=500, step=16),
    #                               kernel_size=21, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Conv1D(filters=364, kernel_size=hp.Int('kernel1_size', min_value=13, max_value=41, step=3),
                                  padding="causal", dilation_rate=2, activation='relu', name="Conv1D_1"))

    model.add(keras.layers.MaxPooling1D(hp.Int('pool1_size', min_value=1, max_value=3, step=1)))

    model.add(keras.layers.Conv1D(filters=126, kernel_size=hp.Int('kernel2_size', min_value=5, max_value=21, step=3),
                                  padding="causal", dilation_rate=2, activation='relu', name="Conv1D_2"))

    model.add(keras.layers.MaxPooling1D(hp.Int('pool2_size', min_value=1, max_value=3, step=1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('filter1_size', min_value=10, max_value=600, step=90), activation='relu', name="Dense_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(hp.Int('filter2_size', min_value=10, max_value=300, step=50), activation='relu', name="Dense_2"))
    model.add(keras.layers.Dense(hp.Int('filter3_size', min_value=10, max_value=600, step=80), activation='relu', name="Dense_3"))
    # We want to pass in 1 because this is the output layer, and we only want to predict one thing which is power
    # Otherwise if we predict 24 hrs ahead for each time step, then we have 96 values that the Dense layer should output
    #model.add(keras.layers.Dense(trainY.shape[1], name="Dense_4"))
    model.add(keras.layers.Dense(1, name="Dense_4"))

    optimizer = tf.keras.optimizers.RMSprop(hp.Float('learning_rate', min_value=0.00005, max_value=0.01, step=0.0001))

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])

    # model.fit(trainX, trainY, epochs=epochs,
    #           validation_split=validation_split, verbose=1)
    model.save('CNN_test2.keras')
    return model

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mae']),
           label='Train')
  plt.plot(history.epoch, np.array(history.history['val_mae']),
           label='Val')
  plt.legend()
  plt.ylim([0, max(history.history['val_mae'])])
  plt.savefig('hist.png')


def remove_night_predictions_one_timestep(testY_zero, i, y_pred):
    zero_indices = []

    temp = testY_zero[:, i, 0]
    # Create a mask for zero values
    zero_mask = (temp == 0)

    # Get the indices where the values are zero
    zero_indices = (np.where(zero_mask)[0])

    y_test_list_day = []
    y_pred_day = []

    # Use fancy indexing to remove rows at zero_indices
    y_test_list_day = np.delete(testY[:, i], zero_indices, axis=0)

    y_pred_day = np.delete(y_pred, zero_indices, axis=0)

    return y_test_list_day, y_pred_day


def main():
    datafile_path = 'station01.pkl'

    # load data
    # cols_to_remove = ['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'station',
    #                   'lmd_hmd_directirrad', 'nwp_hmd_diffuseirrad', 'nwp_pressure']

    # From Y = G(X), where Y is the true data, and X the input. In this case we have X containing NWP data used to make
    # the prediction and Y is the true power data, taken from LMD
    # 24 hrs x 4 timesteps (i.e. sampled at each 15 min, so 4 times per hr)
    # 2 x 4, that is for the past two hrs, 4 samples per hr
    # Each sample takes values for 96 nwp from a particular timestep
    #trainX, trainY, testX, testY = load_CNN_data(datafile_path, cols_to_remove, 24*4, 2*4)
    #trainX, trainY, testX, testY = LS.load_LSTM_zero_padded('station02', 24 * 4, 24 * 4)

    # ________________________This uses Keras build in random search hyperparameter optimizer_______________________
    # Set up the tuner
    tuner = RandomSearch(
        fit_CNN_tune,  # Model-building function
        objective='val_accuracy',  # Metric to optimize
        max_trials=50,  # Number of different hyperparameter combinations to try
        directory='Tuner',  # Directory to save results
        project_name='CNN_hp')  # Project name


    # Search for the best hyperparameters
    #tuner.search(trainX, trainY[0], epochs=3, validation_split=0.2)
    # trainY[:, 1] get all time steps values for only 15 min ahead

    predictions_list = []
    testY_day_list = []
    r2_list = []
    mse_list = []
    i = 95
    for i in range(96):
        tuner.reload()
        # Retrieve the best performing model
        best_model = tuner.get_best_models(num_models=1)
        # tuner.search(trainX, trainY[:, i], epochs=3, validation_split=0.2)

        # Get the best parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=100)[0]

        # Build and train the final model
        final_model = tuner.hypermodel.build(best_hps)
        history = final_model.fit(trainX, trainY[:, i], batch_size=64, epochs=15, validation_split=0.2, verbose=1)

        # Plot the history of training and validation error
        # The further they are from each other, the more the model is over fitting the train data
        # plot_history(history)

        # Evaluate the model
        eval_result = final_model.evaluate(testX, testY[:, i])

        print("Test accuracy:", eval_result[1])
        print("Test mae:", eval_result[0])

        power_predictions = final_model.predict(testX)

        # Remove night data from the predictions
        non_zero_indices = np.nonzero(testY[:, i])[0]
        testY_day = testY[:, i][non_zero_indices]

        power_pred_day = np.squeeze(power_predictions, axis=1)
        power_pred_day = power_pred_day[non_zero_indices]
        power_pred_day = np.expand_dims(np.asarray(power_pred_day), axis=1)

        # testY_day, power_pred_day = remove_night_predictions_one_timestep(testY, i, power_predictions)
        predictions_list.append(power_pred_day)
        testY_day_list.append(testY_day)

        r2 = r2_score(testY[:, i], power_predictions)
        mse = mean_squared_error(testY[:, i], power_predictions)

        r2_day = r2_score(testY_day, power_pred_day)
        mse_day = mean_squared_error(testY_day, power_pred_day)
        r2_list.append(r2_day)
        mse_list.append(mse_day)

        print('R2 score: ', r2)
        print('MSE: ', mse)

        print('R2 score day: ', r2_day)
        print('MSE day : ', mse_day)

    # plt.figure()
    # plt.scatter(testY, power_predictions.flatten())
    # plt.ylabel('Predicted')
    # plt.xlabel('True values')
    # plt.ylim(-0.2, 1)
    # plt.xlim(-0.2, 1)
    # plt.plot([0, 1], [0, 1], color='r')
    # plt.show()

    # n = 0 is predicting 15 min ahead
    # n = 95 is predicting 24 hrs ahead
    n = 0
    plt.figure()
    testY_day_array = np.asarray(testY_day_list)
    testY_day_array = testY_day_array.reshape(testY_day_array.shape[1], testY_day_array.shape[0])
    plt.plot(testY_day_list[95], label='true value')
    predictions_array = np.asarray(predictions_list).reshape(np.asarray(predictions_list).shape[1], np.asarray(predictions_list).shape[0])
    plt.plot(predictions_list[95], label='predicted')
    # plt.plot(testY.reshape(testY.shape[0], testY.shape[1])[:, n], label='true value')
    # plt.plot(power_predictions[:, n], label='predicted')
    plt.legend()
    plt.savefig('Pred_vs_true.png')
    plt.show()

    plt.figure()
    plt.plot(r2_list)
    plt.title('R2 score for day data')
    plt.xlabel('Time steps')
    plt.ylabel('R2')
    plt.savefig('r2_score.png')
    plt.show()

    plt.figure()
    plt.plot(mse_list)
    plt.title('Minimum Squared Error for day data')
    plt.ylabel('MSE')
    plt.xlabel('Time steps')
    plt.savefig('mse.png')
    plt.show()

    print('test')

if __name__ == '__main__':
        main()