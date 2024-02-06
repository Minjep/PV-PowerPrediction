
import os, sys
import pickle
from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
from timezonefinder import TimezoneFinder
import datetime
import logging as log
from sklearn.model_selection import GridSearchCV
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split
# standardscaler
from sklearn.preprocessing import StandardScaler
# pipeline
from sklearn.pipeline import Pipeline
import sklearn as sk
import pandas as pd
from sunsetSunriseAPI import sunrisesunset
def loadDataAndSplit(stationNum,testSize=1000):
    stationData2 = fileLoader.loadFile(f"station0{stationNum}.csv",path=None,PKL=True)
    # if ModelsSVR file exists then load it
    testSize=round(len(stationData2)*0.2)
    stationTrain=stationData2[:-testSize]
    stationTest=stationData2[-testSize:]
    return stationTrain,stationTest
def normalizeFromMetadata(data):
    """
    Normalize data using the metadata file. 
    """
    meta=fileLoader.loadFile("metadata.csv")

    stationNum=data["station"][0]
    meta_station=meta[meta["Station_ID"]==f"station0{stationNum}"]
    # use the capacity of the station to normalize the power
    data["power"]=data["power"]/meta_station["Capacity"][stationNum]
    return data
def loadDataCorrect(data,hoursAhead=24.0):
    data=data.drop(columns=['nwp_pressure','lmd_pressure'])
    # The LMD named features is not known 24 hours in advance, so we shift the data 24 hours back
    for feature in data.columns:
        if feature.startswith("lmd"):
            # theres 4 samples per hour, so we shift 24*4=96 samples back
            # fx if we have lmd at 12:00 we move that measurement to the next day at 12:00
            data[feature]=data[feature].shift(hoursAhead)
    data["powerLast"]=data["power"].shift(hoursAhead)
    # drop the first 96 samples
    data=data.dropna()
    #print(data["lmd_totalirrad"])
    return data
def splitContinuousData(data):
    features=data.drop(columns=['power','date_time'])
    power=data['power']
    # cutout 10% of the data
    lenData=round(len(power)*0.2)
    featuresTest=features[-lenData:]
    features=features[:-lenData]
    powerTest=power[-lenData:] 
    power=power[:-lenData] 
    return features,featuresTest,power,powerTest
def trainModel(model,features,power):
    
    X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.1, random_state=42)
    clonedPipe=sk.clone(model).fit(X_train, y_train)
    return clonedPipe
def gridSearch(features,power):
    K = 25               # Number of cross valiations
    # Parameters for tuning
    s=StandardScaler()
    s1=StandardScaler()
    X = s.fit_transform(features)
    y = s1.fit_transform(np.array(power).reshape(-1,1))
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]
    print("Tuning hyper-parameters")
    svr = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = K)
    svr.fit(X, y.ravel())
    print("Best parameters set found on development set:")
    gamma=svr.best_params_['gamma']
    C=svr.best_params_['C']
    return [gamma, C]
def trainAndGridSearch(features,power):
    [gamma, C]=gridSearch(features,power)
    model_pipe=Pipeline([("scaler",StandardScaler),("svr",SVR(kernel='rbf',gamma=gamma,C=C))])
    return trainModel(model_pipe,features,power)
def saveModels(Models,subfix):
    file = open(f"ModelsSVR{subfix}.pkl", "wb")
    pickle.dump(Models, file)
    file.close()
def train96ModelsLMDAhead(subfix,model,stationTrain):
    Models=[]
    if not os.path.isfile(f"ModelsSVR{subfix}.pkl"):
        for i in range(0,96,1):
            stationData=loadDataCorrect(stationTrain,i)
            [features,featuresTest,power,powerTest]=splitContinuousData(stationData)
            Models.append((trainModel(model,features, power)))
            # calculate the score of the regression model
            score = Models[i].score(featuresTest, powerTest)
            print(f"{i/4} Hours ahead validation score:{score}")
        saveModels(Models,subfix)
def predict96ModelsLMDAhead(Models_,stationTest):
    predictedValues=np.zeros((96,len(stationTest)))
    scorelist=[]
    
    i=0
    for model in Models_:
        stationData=loadDataCorrect(stationTest,i)
        features=stationData.drop(columns=['power','date_time'])
        power=stationData['power']
        score =model.score(features, power)
        scorelist.append(score) 
        predictedValues[i,:]=(np.append(np.zeros((i,1)),model.predict(features)))
        print(f"{i/4} Hours ahead test score:{score}")
        i=i+1
    return predictedValues,scorelist


# Load csv data with night data
# We know NWP features for each power point. 
# The LMD data is known for the last 24 hours at midnight.
# The last 24 hours is flattened, so we add LMD_features*24 hours

def load_data_arrayofdays(timegroup,stationNum=1):
    data=fileLoader.loadFile(f"station0{stationNum}.csv",path=None,PKL=False)
    meta=fileLoader.loadFile("metadata.csv")
    
    data=data.drop(columns=['nwp_pressure','lmd_pressure'])
    # we cut the data into sections of 24 hours starting at midnight
    # we have 4 samples per hour, so we cut the data into sections of 96 samples by the midnight timestamp
    time_column = "date_time"
    # group by the date part of the time column at midnight

    data[time_column] = pd.to_datetime(data[time_column])
    # convert the time column to datetime format
    # group the dataframe by the date part of the time column
    # group by the time set as parameter
    # get location of station from meta data
    meta_station=meta[meta["Station_ID"]==f"station0{stationNum}"]
    meta_station=meta_station.reset_index()
    st_longitude = meta_station["Longitude"][0]
    st_latitude = meta_station["Latitude"][0]
    # use the location to calculate when midnight is in UTC
    tf = TimezoneFinder()
    tz1 = tf.timezone_at(lng=st_longitude, lat=st_latitude)
    # calculate the time difference between UTC and the station
    import pytz
    timezone_station=pytz.timezone(tz1)
    # get the time difference as integer between UTC and the station
    time_difference=timezone_station.utcoffset(datetime.datetime.now()).total_seconds()/3600
    
    
    
   
    timestamp=pd.Timestamp(timegroup+time_difference, unit='h')

    groups = data.groupby(pd.Grouper(key=time_column, freq="D",origin=timestamp))
    # create an empty list to store the sub-dataframes
    result = []
    # loop through the groups and append each group to the list
    for _, group in groups:
        if not group.empty:
            result.append(group)
    # return the list of sub-dataframes
    
    return result
def load_data_lmd_flatten(data_grouped, historyOfLMD=24):
    # cut take all the lmd data from the previous 0-24 hours and flatten it into one row
    data_groupednew=data_grouped.copy()
    percentStatus=0.0
    for dayNum in range(2,len(data_grouped)-1):
        new_lmd_data=pd.DataFrame()
        names=data_grouped[dayNum].columns[data_grouped[dayNum].columns.str.startswith('lmd')]
        lmd_data=data_grouped[dayNum-1][data_grouped[dayNum].columns[data_grouped[dayNum].columns.str.startswith('lmd')]]
        # keep the last x points of the lmd data
        lmd_data=lmd_data.tail(historyOfLMD*4)
        # add the columns to the dataframe
        new_columns = pd.DataFrame(
            {
                f"{col}_{row[0].strftime('%T')}": row[1][col]
                for row in lmd_data.iterrows()
                for col in lmd_data.columns
            },
            index=[data_grouped[dayNum].iloc[0]["date_time"]],
        )
        # concatenate the new DataFrame with the original one
        data_groupednew[dayNum]=data_groupednew[dayNum].drop(columns=names)

        data_groupednew[dayNum] = pd.concat([data_groupednew[dayNum], new_columns], axis=1)
        # fill the nan with the previous data
        data_groupednew[dayNum]=data_groupednew[dayNum].ffill()
        percentStatusNow=round(dayNum/len(data_grouped)*100,2)
        if percentStatusNow>percentStatus+1:
            percentStatus=percentStatusNow
            log.debug(f"Flattening {percentStatus}% done")
        # add the new data to the dataframe and drop the old data
        # 
        #data_groupednew[dayNum]=pd.concat([nonLMD_data,new_lmd_data],axis=1)
        #print(data_groupednew[dayNum].head())
        #print(data_groupednew[dayNum].columns)
    return data_groupednew
def removeNightData(station_data,stationNum=1):
    meta = fileLoader.loadFile("metadata.csv",path=None,PKL=False)

    f = r"https://api.sunrise-sunset.org/json?"

    count = 0
    # Initialize an non existing date
    previous_date = '0000-00-00'
    sunrise_sunset = 0.0, 0.0
    # Loop through the entire data from all stations
    station_data["date_time_new"] = pd.to_datetime(station_data.index)
    
    totalValues=len(station_data.values)
    statusPercent=0.0
    indicesMask=[]
    for i, data in enumerate(station_data.values):
        # Find latitude and longitude from metadata variable based on station number stored in data[15]
        longitude = meta["Longitude"][stationNum]
        latitude = meta["Latitude"][stationNum]
        # Get the date of the measurement
        current_date = data[-1].date()

        # Call the API per day and not per each measurement, to save computations time
        if previous_date != current_date:
            previous_date = current_date

            params = {"lat": latitude, "lng": longitude, "date": current_date}
            # Use the API to return sunrise and sunset times based on the three parameters
            sunrise_sunset = sunrisesunset(f, params)

        # keep only data from sunrise til sunset, i.e. remove night data
        # use both hour and minutes
        current_time = float(str(data[-1].time().hour)+"." + str(data[-1].time().minute))
        sunrise_time = float(sunrise_sunset[0].split(':')[0] + '.' + sunrise_sunset[0].split(':')[1])
        sunset_time = float(sunrise_sunset[1].split(':')[0] + '.' + sunrise_sunset[1].split(':')[1])
        # Check if current time is not in between sunrise (PM) and sunset (AM) because of UTC
        if sunrise_time > current_time > sunset_time:
            # remove row i based on the Timestamp index
            # Make sure to run this only one station at a time, since there are the same
            # timestamps for each station, but located at different latitudes & longitudes
            # resulting in different sunset & sunrise
            indicesMask.append(i)
            #station_data.drop(station_data.index[i-count], inplace=True)
            count = count + 1
        newStatusPercent=round(i/totalValues*100,2)
        if newStatusPercent>statusPercent+1:
            statusPercent=newStatusPercent
            log.debug(f"Removing night data {newStatusPercent}% done")
    station_data=station_data.drop(columns=['date_time'])
    station_data=station_data.drop(station_data.index[indicesMask])
    return station_data
if __name__ == "__main__":
    sharedFolder=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project"
    stationTrain,stationTest=loadDataAndSplit(1)

    #create_sequences2(stationTrain, 24, 4)
    #create_sequences(stationTrain, 24, 4)

    PipeRadial=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf'))])
    # Train 96 models
    train96ModelsLMDAhead("grid",PipeRadial,stationTrain)
    # load models generated
    ModelsRbf=pickle.load(open(f"ModelsSVR{'12-15'}.pkl", "rb"))
    # predict on the 96 models
    print("Models loaded, predicting...")
    predictedValuesrbf,scorelistrbf=predict96ModelsLMDAhead(ModelsRbf,stationTest)
    # Plot results
    plt.figure(figsize=(16*0.75,9*0.75))
    plt.plot(stationTest['power'].values[3*96+3:7*90],label="Actual")
    plt.plot(predictedValuesrbf[1,3*96+3:7*90],label="Predicted 15 minutes ahead every 15'th minute")
    plt.plot(predictedValuesrbf[95,3*96+3:7*90],label="Predicted 24 hours ahead every 15'th minute")
    plt.xlabel("Time [UTC]",fontsize=14)
    plt.ylabel("Power [MW]",fontsize=14)
    # calculate the xticks from stationTest index
    dataLength=len(stationTest.index[3*96+3:7*90])
    numberOfTicks=6
    rangeOfLabels=stationTest.index.date[0:dataLength:dataLength//numberOfTicks]
    if len(rangeOfLabels)>numberOfTicks:
            rangeOfLabels=stationTest.index.date[0:dataLength:(dataLength//numberOfTicks)+1]
    if len(rangeOfLabels)<numberOfTicks:
            rangeOfLabels=stationTest.index.date[0:dataLength:(dataLength+1//numberOfTicks)-1]
    plt.xticks(np.linspace(0,dataLength,numberOfTicks),rangeOfLabels,rotation=90,fontsize=14)
    plt.yticks(fontsize=14)
        
    # set legend fontsize
    plt.legend(fontsize=12,loc="upper right")
    plt.title("SVR prediction of power at different time horizons",fontsize=20)
    plt.tight_layout()
    plt.savefig(sharedFolder+r"\Figures\15and30minPrediction.png",format="png",bbox_inches='tight')
    # save file with models
    #plt.savefig(sharedFolder+r"\Figures\SVR_powerSingleLMD.png",format="png",bbox_inches='tight')
    # scores to from 1-8 quarters ahead
    from sklearn.metrics import mean_squared_error  
    SVRmse=np.zeros(96)
    for i in range(96):
        SVRmse[i] = mean_squared_error(stationTest['power'].values, predictedValuesrbf[i,:])
        

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    # Plot MSE
    time_steps_hours = np.arange(0, 24, 0.25)
    ax1.scatter(time_steps_hours, SVRmse, marker='o', linestyle='-', label='SVR')
    ax1.set_title('Mean Squared Error (MSE)',fontsize=20)
    ax1.set_xlabel('Time into the future (hours)',fontsize=16)
    ax1.set_ylabel('MSE',fontsize=16)
    ax1.legend(fontsize=12)
    # Plot R-squared
    ax2.scatter(time_steps_hours, scorelistrbf, marker='o', linestyle='-', label='SVR')
    ax2.set_title('R-squared',fontsize=20)
    ax2.set_xlabel('Time into the future (hours)',fontsize=16)
    ax2.set_ylabel('R-squared',fontsize=16)
    ax2.legend(fontsize=12)
    # Adjust layout for better readability
    plt.tight_layout()
    # Show the plots

    plt.savefig(sharedFolder+r"\Figures\SVR_powerSingleLMD21_score.png",format="png",bbox_inches='tight')
    # save to a file
    
    print("Saving to file...")
    with open(sharedFolder+r"\Figures\SVR96Predictor2_st1.pkl", "wb") as f:
        dictofData={
            "TrainingData":stationTrain,
            "TestData":stationTest,
            "Models":ModelsRbf,
            "PredictedValues":predictedValuesrbf,
            "Scores":scorelistrbf
        }
        pickle.dump(dictofData, f)