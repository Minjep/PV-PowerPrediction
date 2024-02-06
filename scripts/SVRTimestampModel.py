# Script to either create a new model or predict using an existing model
# The model tries to predict a day ahead from a given time of the day
import SVRMachineLearning 
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle   
import matplotlib.pyplot as plt
import os

import logging as log

def load_data_station_time(station, time,lmdHours=1):
    log.debug("Loading data for station %s known at time %s", station, time)
    data = SVRMachineLearning.load_data_arrayofdays(time,station)
    log.debug("Data loaded, adding LMD and flattening")
    data = SVRMachineLearning.load_data_lmd_flatten(data,lmdHours)
    # drop the columns that are not complete
    log.debug("Dropping incomplete rows")
    data=data[2:-1]
    data=pd.concat(data)
    data=data.dropna()
    
    colsDelete=[]
    for i in data.columns:
        if "lmd" in i:
            colsDelete.append(i)
    data=data.drop(columns=colsDelete)
    data["previous_power"]=data["power"].shift(1)
    data["timeNum"]=data["date_time"].apply(lambda x: x.time().hour)
    # drop the first row as it is NaN
    data=data.dropna()
    
    log.debug("Data structure complete, removing night data")
    data=SVRMachineLearning.removeNightData(data,station)
    log.info("Data loaded for station %s known at time %s", station, time)
    return data
def split_data_train_test(data):
    log.info("Splitting data into train and test")
    cols=data.columns.tolist()
    colsDelete=["power"]
    if "date_time" in cols:
        log.debug("Dropping date_time column")
        colsDelete.append("date_time")
    if 'date_time_new' in cols:
        log.debug("Dropping date_time_new column") 
        colsDelete.append("date_time_new")
    
    features=data.drop(columns=colsDelete)
    power=data['power']
    # cutout 20% of the data
    lenData=round(len(power)*0.2)
    featuresTest=features[-lenData:]
    features=features[:-lenData]
    powerTest=power[-lenData:] 
    power=power[:-lenData] 
    log.info(f"Data split into train and test,{lenData} is the length of the test data")
    return features,featuresTest,power,powerTest
def train_and_save_model(features,features_test,power,power_test,station,time):  
    log.info("Training model")
    log.debug("Creating pipeline")  
    Pipe=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
    log.debug("Fitting pipeline")
    Pipe.fit(features,power)
    log.debug("Saving model")
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'wb') as file:
        dictOfData={'station':station,
                    'time':time,
                    'model':Pipe,
                    'train_features':features,
                    'train_power':power,
                    'test_features':features_test,
                    'test_power':power_test}
        pickle.dump(dictOfData,file)
    log.info("Model saved as svr_t%s_st%s.pkl",time,station)
    
def predict_and_save(station,time):
    log.info("Loading model")
    # check if the model exists
    dictOfData=load_model(station,time)
    if dictOfData is None:
        log.error("Model does not exist")
        return
    log.debug("Model loaded")
    model=dictOfData['model']
    features=dictOfData['test_features']
    log.info("Predicting")
    last_Power=0
    predictions=[]
    # loop rows in the test data
    i=0
    for index,row in features.iterrows():
        # add a column with the previous power
        # predict the power
        features.loc[index,"previous_power"]=last_Power# set the previous power to the last predicted power
        features.loc[index,"timeNum"]=index.hour
        predicted_power=model.predict(features.iloc[i:i+1]) # 
        last_Power=predicted_power
        predictions.append(predicted_power)
        i+=1
    log.info("Appending prediction to pickle file")
    dictOfData['prediction']=predictions
    # overwrite the file
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'wb') as file:
        pickle.dump(dictOfData,file)
    log.info("Prediction saved to pickle file")
    return predictions
def load_model(station,time):
    if not os.path.exists(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl"):
        log.error("Model does not exist")
        return
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'rb') as file:
        dictOfData=pickle.load(file)
    log.debug("Model loaded")
    return dictOfData
    
def calculate_scores(station,time):
    log.info("Scoring model")
    # load the model
    dict_of_data=load_model(station,time)
    if dict_of_data is None:
        log.error("Model does not exist")
        return 0,0
    if not dict_of_data.keys().__contains__('prediction'):
        log.info("Model does not contain prediction, predicting..")
        predict_and_save(station,time)
        dict_of_data=load_model(station,time)

    power=dict_of_data['test_power']
    prediction=dict_of_data['prediction']
    # calculate the score
    score_r2=r2_score(power,prediction)
    score_mse=mean_squared_error(power,prediction)
    return score_r2,score_mse
def get_value_from_model(station,time,value):
    log.info("Getting value from model")
    # load the model
    dict_of_data=load_model(station,time)
    if dict_of_data is None:
        log.error("Model does not exist")
        return
    if dict_of_data.keys().__contains__(value):
        log.debug("Value %s found in model",value)
        return dict_of_data[value]
    else:
        log.error("Model does not contain value %s",value)
    return None

def simple_example(station,time):
    # set the station and time

    # load the data
    data=load_data_station_time(station,time)
    # split the data
    features,features_test,power,power_test=split_data_train_test(data)
    # train the model
    features, power = shuffle(features, power, random_state=42)
    train_and_save_model(features,features_test,power,power_test,station,time)
   
    
def example_with_existing_model(station,time):
    # set the station and time
    # predict using the model
    prediction=predict_and_save(station,time)
    # get the score
    score_r2,score_mse=calculate_scores(station,time)
    # get a value from the model
    true_power=get_value_from_model(station,time,'test_power')
    
def example_checking_if_model_exists(time,station):
    log.info("Checking if model exists")
    if os.path.exists(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl"):
        log.info("Model exists")
        example_with_existing_model(station,time)
    else:
        log.info("Model does not exist")
        simple_example(station,time)
        example_with_existing_model(station,time)
def multiple_models():
    for station in range(1,8):
        for timeStamp in [0,3,6,9,12,15,18]:
            simple_example(station,timeStamp)
            
def plotPrediction(station,time):
    # load the data
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
    # plot the predictions
    testy=data['test_power']
    predictions=data['prediction']
    print("Score r2 of data0:",r2_score(testy,predictions))
    plt.figure(figsize=(16,9))
    
    plt.plot(testy.values[1270:1450],label="True power")
    plt.plot(predictions[1270:1450],label="Prediction")
    # set the x axis to have dates as ticks
    plt.xticks(np.linspace(0,len(testy.values[1270:1450]),4),testy.index.date[1270:1450:len(testy.values[1270:1450])//4],rotation=90)
    #plt.locator_params(axis='x', nbins=10)
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Power [kW]",fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.title(f"Model with feedback power for station {station} : R2 score = {r2_score(testy,predictions):.2f}",fontsize=20)
    plt.savefig(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\feedBackPowerModel.png",bbox_inches='tight')
           
def main():
    # set the logging level
    log.basicConfig(level=log.DEBUG)
    log.info("Starting script")
    plotPrediction(1,12)
    r2,_=calculate_scores(1 ,12)
    print(r2)
    
    simple_example(1,12)
    example_with_existing_model(1,12)
    multiple_models()
    log.info("Script done")
if __name__ == "__main__":
    main()