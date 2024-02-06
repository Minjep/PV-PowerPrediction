from matplotlib import pyplot as plt
import pickle
import os, sys
import numpy as np
import pandas as pd
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PV_PredictLib import LSTM
from PV_PredictLib import fileLoader as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Load the DataFrames from pickle files
df=pd.read_pickle(r'C:\Users\jeppe\gits\PV-PowerPrediction\plotScripts\all_fixed_data.pkl')
# set the order of the columns
df=df[['Actual Power', 'pred_svr', 'pred_lr', 'pred_lstm', 'pred_cnn']]
# rename the columns to be more descriptive
df.columns=['Actual Power', 'SVR Predictions', 'LR Predictions', 'LSTM Predictions', 'CNN Predictions']
# remove the rows where the actual power is 0
df_day = df[df['Actual Power'] != 0]
# build SVR and LR dataframe
df_svr_lr = df_day[['Actual Power', 'SVR Predictions', 'LR Predictions']]
df_lstm_cnn = df_day[['Actual Power', 'LSTM Predictions', 'CNN Predictions']]
# Calculate scores for some reason
cnn_mse=mean_squared_error(df_day['Actual Power'],df_day['pred_cnn'])
cnn_r2=r2_score(df_day['Actual Power'],df_day['pred_cnn'])

# Select timeslices for plotting good and bad days
start_time = pd.to_datetime('2019-05-01 16:00:00')
end_time = start_time + pd.Timedelta(days=2) 
# Good days
filtered_svr_lr_good = df_svr_lr.loc[start_time:end_time]
filtered_lstm_cnn_good = df_lstm_cnn.loc[start_time:end_time]
# Bad days
start_time = pd.to_datetime('2019-04-23 16:00:00')
end_time = start_time + pd.Timedelta(days=2)
filtered_svr_lr_bad = df_svr_lr.loc[start_time:end_time]
filtered_lstm_cnn_bad = df_lstm_cnn.loc[start_time:end_time]
# do a function describing how to use the function

def plotFixedPointComparison(data,numberOfTicks,fileSaveName,title="24 hours prediction"):
    '''
    A function that plots the data and saves it as a file
        data: dataframe with the data to plot
        numberOfTicks: number of ticks on the x axis
        fileSaveName: name of the file to save the plot as
        title: title of the plot
    '''
    # sæt figur størrelse
    dataValues=data.values
    dataLength=len(dataValues)
    plt.figure(figsize=(16*0.75,9*0.75))
    plt.plot(dataValues)
    # divide the x axis into NumberOfTicks
    dataValues=data.values
    dataLength=len(dataValues)
    rangeOfLabels=data.index[0:dataLength:dataLength//numberOfTicks]
    if len(rangeOfLabels)>numberOfTicks:
        rangeOfLabels=data.index[0:dataLength:(dataLength//numberOfTicks)+1]
    if len(rangeOfLabels)<numberOfTicks:
        rangeOfLabels=data.index[0:dataLength:(dataLength+1//numberOfTicks)-1]
    plt.xticks(np.linspace(0,dataLength,numberOfTicks),rangeOfLabels,rotation=90)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Power [Mega Watt]')
    plt.title(title,fontsize=20)
    plt.legend(data.columns,loc='center right', bbox_to_anchor=(1.2, 0.5))
    # plot only the
    plt.tight_layout()
    plt.savefig(fileSaveName,bbox_inches='tight')
    
# plot good days
plotFixedPointComparison(filtered_svr_lr_good,6,'clear_svr_lr.png','24 hours prediction on clear sky days')
plotFixedPointComparison(filtered_lstm_cnn_good,6,'clear_lstm_cnn.png','24 hours prediction on clear sky days')

# plot bad days
plotFixedPointComparison(filtered_svr_lr_bad,6,'cloudy_svr_lr.png','24 hours prediction on cloudy days')
plotFixedPointComparison(filtered_lstm_cnn_bad,6,'cloudy_lstm_cnn.png','24 hours prediction on cloudy days')
