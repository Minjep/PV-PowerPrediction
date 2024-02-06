# SVR model to predict 24 hours ahead of power production
# By feeding back the the last power prediction

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA 
import pickle

# import homemade
import SVRMachineLearning as SVRML
import SVRTimestampModel as SVRTM
def addPreviousPowerToTrainingData(data):
    # add the previous power to the training data
    #data=pd.concat(data)
    # remove all lmd
    colsDelete=[]
    for i in data.columns:
        if "lmd" in i:
            colsDelete.append(i)
    data=data.drop(columns=colsDelete)
    data["previous_power"]=data["power"].shift(1)
    data["timeNum"]=data["date_time"].apply(lambda x: x.time().hour)
    # drop the first row as it is NaN
    data=data.dropna()
    return data
def addPreviousPowerToSplittedTrainingData(feature,power):
    feature["previous_power"]=power.shift(1)
    feature["timeNum"]=feature.index.hour
    # drop the first row as it is NaN
    data=feature.dropna()
    # drop the same rows from the power
    power=power.drop(data.index[0])
    return data,power


def trainFeedBackPowerModel(station,time):
    # load the data
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
    
    
    # split the data into train and test
    trainX,trainY = addPreviousPowerToSplittedTrainingData(data['train_features'],data['train_power'])
    testX = data['test_features']
    colsDelete=[]
    #for i in trainX.columns:
    #    if "lmd" in i:
    #        colsDelete.append(i)
    trainX=trainX.drop(columns=colsDelete)
    testX=testX.drop(columns=colsDelete)

    testY = data['test_power']
    # create the pipeline
    Pipe=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
    
    Pipe.fit(trainX,trainY)

    last_Power=0
    predictions=[]
    # loop rows in the test data
    i=0
    for index,row in testX.iterrows():
        # add a column with the previous power
        # predict the power
        testX.loc[index,"previous_power"]=last_Power# set the previous power to the last predicted power
        testX.loc[index,"timeNum"]=index.hour
        predicted_power=Pipe.predict(testX.iloc[i:i+1]) # 
        last_Power=predicted_power
        predictions.append(predicted_power)
        i+=1
        # add the predicted power to the features
    # save a pickle with model and data
    
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svrFB_t{time}_st{station}.pkl", 'wb') as file:
            dictOfData={'model':Pipe,
                        'train_features':trainX,
                        'train_power':trainY,
                        'test_features':testX,  
                        'test_power':testY,
                        'predictions':predictions}
            pickle.dump(dictOfData,file)
    print("Done")
    print("Score r2 of data0:",r2_score(testY,predictions))  
def plotDayAhead(station,time):
    # load the data
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svrFB_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
    # plot the predictions
    plt.figure(figsize=(16*0.75,9*0.75))
    
    plt.plot(data["test_power"].values[1270:1450],label="True power")
    plt.plot(data["predictions"][1270:1450],label="Prediction")
    # set the x axis to have dates as ticks
    plt.xticks(np.linspace(0,len(data["test_power"].values[1270:1450]),4),data["test_power"].index.date[1270:1450:len(data["test_power"].values[1270:1450])//4],rotation=90,fontsize=14)
    plt.yticks(fontsize=14)
    #plt.locator_params(axis='x', nbins=10)
    plt.xlabel("Time [UTC]",fontsize=18)
    plt.ylabel("Power [MW]",fontsize=18)
    plt.legend(location='upper right',fontsize=14)
    plt.title(f"Feedback model for station {station} : R2 score = 0.82",fontsize=20)
    plt.tight_layout()
    print("Score r2 of data0:",r2_score(data['test_power'],data['predictions']))
    plt.tight_layout()  
    plt.savefig(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\DayAheadprediction{time}.png",bbox_inches='tight')
def trainSimpleModel(station,time):
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svrFB_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
    # split the data into train and test
    trainX=data['train_features']
    testX = data['test_features']
    trainy=data['train_power']
    testy=data['test_power']
    colsDelete=['nwp_windspeed', 'nwp_winddirection','timeNum','previous_power']
    for i in trainX.columns:
        if "lmd" in i:
            colsDelete.append(i)
    trainX=trainX.drop(columns=colsDelete)
    testX=testX.drop(columns=colsDelete)
    # create the pipeline
    Pipe=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
    Pipe.fit(trainX,trainy)
    predictions=Pipe.predict(testX)
    print("Score r2 of data0:",r2_score(testy,predictions))
    plt.figure(figsize=(16*0.75,9*0.75))
    
    plt.plot(testy.values[1270:1450],label="True power")
    plt.plot(predictions[1270:1450],label="Prediction")
    # set the x axis to have dates as ticks
    plt.xticks(np.linspace(0,len(testy.values[1270:1450]),4),testy.index.date[1270:1450:len(testy.values[1270:1450])//4],rotation=90,fontsize=14)
    plt.yticks(fontsize=14)
    #plt.locator_params(axis='x', nbins=10)
    plt.xlabel("Time [UTC]",fontsize=18)
    plt.ylabel("Power [MW]",fontsize=18)
    plt.legend(location='upper right',fontsize=14)
    plt.title(f"Simple model for station {station} : R2 score = 0.78",fontsize=20)
    plt.tight_layout()

    plt.savefig(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\simpleModel.png",bbox_inches='tight')
    
def main():
    # load the data
    for i in range(0,8,1):
        trainFeedBackPowerModel(1,0)
def plotScoreVsTime(station,time):
    
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svrFB_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
        print(data.keys())
    # split the data into train and test
    testX = data['test_features']
    testy=data['test_power']
    predictions=data['predictions']
    scores=[]
    for i in range(0,24,1):
        # row number where testX time is i 
        # create list of 0 to length of testX
        listX=list(range(0,len(testX)))
        # set index of testX to listX
        testX.index=listX
        # get the rows where the time is i
        testXTime=testX[testX["timeNum"]==i].index
        
        
        newTesty=testy[testXTime]
        newPredictions=np.concatenate( predictions, axis=0 )[testXTime]
        # calculate the score
        if len(newTesty)==0:
            scores.append(0)
            continue
        score=r2_score(newTesty,newPredictions)
        scores.append(score)
        
    plt.figure(figsize=(16,9))
    plt.bar(range(0,24,1),scores)
    plt.xlabel("Time")
    plt.ylabel("R2 Score")
    plt.title(f"R2 score for station {station} for different times",fontsize=20)
    plt.savefig(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\R2Score.png",bbox_inches='tight')
    plt.show()
if __name__=="__main__":
    trainSimpleModel(1,0)
    #plotScoreVsTime(1,0)

    plotDayAhead(1,0)
