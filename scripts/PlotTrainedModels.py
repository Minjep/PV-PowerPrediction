# Plot the the scores of the trained time models

import pickle
from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
import logging as log
from sklearn.svm import SVR
import datetime as datetime
import SVRTimestampModel
from sklearn.metrics import r2_score,mean_squared_error 
# shared folder
sharedFolder=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project"
# Load the data
times=[0,3,6,9,12,15,18]
station=0
scores=[]
print(SVRTimestampModel.calculate_scores(1,0))
data0=SVRTimestampModel.load_model(1,times[0])
data3=SVRTimestampModel.load_model(1,times[4])
data2=SVRTimestampModel.load_model(1,times[2])

def plotDayAheadWithLineOfPrediction(ax,data,sliceStart,sliceEnd):
    time = datetime.datetime(2020, 1, 1, data["time"], 0, 0).strftime("%H:%M")
    ax.plot(data['test_power'][sliceStart:sliceEnd].values,".-",label="True power",color='C0')
    ax.plot(data['prediction'][sliceStart:sliceEnd],".-",label="Prediction from "+time,color='C1')
    # a red line indicating the time 9:00 
    timeConverted=(datetime.datetime(2020,1,1,data["time"])-datetime.timedelta(hours=8)).time().hour
    
    #for date in range(0,len(data['test_power'].index[sliceStart:sliceEnd])):
        #if data['test_power'].index[sliceStart+date].hour==timeConverted and data['test_power'].index[sliceStart+date].minute==0:
            #ax.axvline(x=date,color='r',linestyle='--')
        
    # add a single label that is used for both lines in the legend
    #ax.plot([], [], color='r',linestyle='--', label="Prediction time "+time)
    ax.set_xticks(np.linspace(0,len(data['test_power'].values[sliceStart:sliceEnd]),4),data['test_power'].index.date[sliceStart:sliceEnd:len(data['test_power'].values[sliceStart:sliceEnd])//4],rotation=90,fontsize=14)
    # set the font size of the y axis
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel("Power[MW]",fontsize=18)
    # set less ticks for the x axis
    #ax.set_xticks(data['test_power'].index[sliceStart:sliceEnd])
    #ax.locator_params(axis='x', nbins=3)      
    ax.set_xlabel("Time[UTC]",fontsize=18)
    
    ax.set_title(f"Comparison of actual and predicted power at {time} for three Consecutive Days ",fontsize=18)   
    
    # set the legend in the center right of the plot
    ax.legend(loc='upper right',fontsize=12)
   
    
    
fig,ax=plt.subplots(1,1,figsize=(16*0.75,9*0.75))
plotDayAheadWithLineOfPrediction(ax,data0,1270,1450)
plt.tight_layout()
t=data0["time"]
plt.savefig(sharedFolder+rf"\Figures\NEWprediction{t}.png",bbox_inches='tight')
#score r2 of data0
r2,mse=SVRTimestampModel.calculate_scores(1,18)
print("Score r2 of data0:",r2)
#plt.figure(figsize=(16,9))
#plt.plot(data3['test_power'].values,label="True power")
#plt.plot(data3['prediction'],label="Prediction")
#plt.xlabel("Time")
#plt.legend()
#plt.show()

for i in times:
    r2,mse=SVRTimestampModel.calculate_scores(1,i)
    scores.append(r2)   
# plot the scoresr
plt.figure()
plt.bar(times,scores)
plt.xlabel("Time")
plt.ylabel("R2 Score")
plt.title("R2 Score for each predicition time")
plt.xticks(times)
plt.savefig(sharedFolder+r"\Figures\ScorePlot.png")
# residusal plot showing the score error difference from the 0 time model
plt.figure(figsize=(16*0.75,9*0.75))
plt.plot(times,scores-scores[0],"o-")
plt.xlabel("Time [hour]",fontsize=16)
plt.ylabel("R2 Score difference",fontsize=16)
timeslabel=["00:00","03:00","06:00","09:00","12:00","15:00","18:00"]
plt.xticks(times,timeslabel,fontsize=14)
plt.yticks(fontsize=14)
plt.title("Model score difference from prediction at midnight",fontsize=20)
plt.grid()
plt.savefig(sharedFolder+r"\Figures\scoreDiff.png",bbox_inches='tight')

# plot prediction at midnight and 12
plt.figure(figsize=(16,9))
plt.plot(data0['test_power'].values[48:140+48],label="True power")
plt.plot(data0['prediction'][0:140+48],label="Prediction")
plt.plot(data3['prediction'][0:140],label="Prediction 12")
# remove the line between two days
plt.xlabel("Time")
plt.legend()
plt.title("Prediction difference from midnight")
plt.savefig(sharedFolder+r"\Figures\predictionDiff.png",bbox_inches='tight')
plt.show()
predictions=np.zeros([96,100])
trueValues=np.zeros([96,100])

scores=np.zeros([96,7])
log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
for time in times:
    SVRTimestampModel.predict_and_save(1,time)
    data_mod=SVRTimestampModel.load_model(1,time)
    
    for n in range(len(data_mod["test_power"].values)):
        # get the date of the current index
        date=data_mod["test_power"].index[n]
        index96=date.hour*4+date.minute/15
        index96=int(index96)
        # get the prediction for the current index
        pred=data_mod["prediction"][n]
        # get the true power for the current index
        testY=data_mod["test_power"].values[n]
        # calculate the score
        j=0
        while(predictions[index96][j]!=0):
            j+=1
            if j>98:
                pass
        if j==98:
            break
        predictions[index96][j]=pred
        trueValues[index96][j]=testY

    # find the mean of each time
    for i in range(1,96):
        j=0
        while(predictions[i][j]!=0):
            j+=1
            if j>99:
                break
        if j>0:

            scores[i,int(time/3)]=mean_squared_error(trueValues[i][:j],predictions[i][:j])
            
# plot the scores
plt.figure(figsize=(16,9))
plt.plot(np.roll(scores,8*4,axis=0))
plt.xlabel("Time")
plt.ylabel("R2 Score")
plt.legend([12,18])
# x ticks as hours
plt.xticks(np.arange(0, 96, step=4),np.arange(0,24))

plt.tight_layout()  # Adjust layout for better spacing
plt.show()



