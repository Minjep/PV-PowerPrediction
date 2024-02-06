# in this script we plot the data uncertanty by comparing the nwp irradiance with the power
# to see if the nwp irradiance is the reason for the uncertanty

# import libs
from tkinter import font
from matplotlib.font_manager import font_scalings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
# scaler
from sklearn.preprocessing import MinMaxScaler
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import homemade
import PV_PredictLib.fileLoader as fl
import PV_PredictLib.LSTM as LSTM

# load the data
data=pd.read_pickle(r'C:\Users\jeppe\gits\PV-PowerPrediction\plotScripts\all_fixed_data.pkl')
# cut the last 20% of the data
Scaler=MinMaxScaler()
# scale the data
# load the data
dataRaw=fl.loadPkl(f"station01.pkl")    
# find the index of the dataRaw power that match the data power
dataRaw2=dataRaw.loc[data.index[0]:data.index[-1]]
# only keep the indexes that are in the data
datanew=data.drop(data.index.difference(dataRaw2.index))
# drop date_time
dataRaw2=dataRaw2.drop(columns=["date_time"])
datanew=Scaler.fit_transform(datanew)
dataRawNew=Scaler.fit_transform(dataRaw2)
datanew=pd.DataFrame(datanew,columns=data.columns)
dataRawNew=pd.DataFrame(dataRawNew,columns=dataRaw2.columns)
powerOriginal=dataRawNew["power"]
NWP=dataRawNew["nwp_globalirrad"]

PredictLinear=datanew["pred_lr"]
PredictSVR=datanew["pred_svr"]
PredictLSTM=datanew["pred_lstm"]
PredictCNN=datanew["pred_cnn"]

plt.figure(figsize=(16*0.75,9*0.75))
plt.title("Comparison of NWP Global Irradiance and power",fontsize=20)
plt.plot(powerOriginal.values,label="Actual Power")
plt.plot(NWP,label="NWP Global irradiance")
plt.plot(PredictSVR.values,label="SVR Regression")
plt.plot(PredictLSTM.values,label="LSTM Regression")

plt.xlim(400,550)
plt.xlabel("Time",fontsize=13)
plt.ylabel("Power Normalized",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend(fontsize=12)
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_comparison.png",format="png")

# plot the data
[fig,ax]=plt.subplots(2,1,figsize=(16*0.75,9*0.75),sharex=True)
ax[0].plot(powerOriginal.values-NWP.values,label="Difference between power and NWP Global Irradiance")

ax[1].plot(powerOriginal.values-PredictSVR.values,label="Difference between power and SVR prediction")
ax[1].plot(powerOriginal.values-PredictLSTM.values,label="Difference between power and LSTM prediction")
ax[1].plot(powerOriginal.values-PredictCNN.values,label="Difference between power and CNN prediction")

ax[0].grid()
ax[1].grid()
ax[0].set_ylim(-1,1)
ax[1].set_ylim(-1,1)

# link the x axis
# set the x axis to be the same
ax[0].set_xlim(400,550)
plt.suptitle("Comparison of power deviations in NWP Global Irradiance and Predictions",fontsize=20)
plt.xlabel("Sample number",fontsize=13)
#plt.ylabel("Difference in normalized power",fontsize=16)
#plt.xlabel("Time",fontsize=16)
#plt.ylabel(r"$\alpha normalized power",fontsize=13)
ax[0].set_ylabel("Scaled power difference",fontsize=13)
ax[1].set_ylabel("Scaled power difference",fontsize=13)

ax[0].tick_params(axis='y', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)
ax[0].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='x', labelsize=13)
ax[0].legend(loc='upper left',fontsize=12)
ax[1].legend(loc='upper left',fontsize=12)

plt.tight_layout()
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_uncertanty.png",format="png")
plt.show()