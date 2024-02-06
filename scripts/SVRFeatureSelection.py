
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import pickle
import SVRTimestampModel 

def addPreviousPowerToSplittedTrainingData(feature,power):
    feature["previous_power"]=power.shift(1)
    feature["timeNum"]=feature.index.hour
    # drop the first row as it is NaN
    data=feature.dropna()
    # drop the same rows from the power
    power=power.drop(data.index[0])
    return data,power

def runPredict(testX,Pipe):
    predictions=[]
    i=0
    last_Power=0
    for index,row in testX.iterrows():
        # add a column with the previous power
        # predict the power
        testX.loc[index,"previous_power"]=last_Power# set the previous power to the last predicted power
        testX.loc[index,"timeNum"]=index.hour
        predicted_power=Pipe.predict(testX.iloc[i:i+1]) # 
        last_Power=predicted_power
        predictions.append(predicted_power)
        i+=1
    return predictions

def main():
    station=1
    time=9
    # load the data
    with open(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\svr_t{time}_st{station}.pkl", 'rb') as file:
        data=pickle.load(file)
    
        
    # split the data into train and test
    trainX,trainY = addPreviousPowerToSplittedTrainingData(data['train_features'],data['train_power'])
    testX = data['test_features']
    testY = data['test_power']
    # create the pipeline
    
    print("Permutation")
    # loop through smaller subsets of features
    i_old=0
    # loop that draws 10 random numbers from 0 to the number of features
    # then takes the features from the training data and fits the model
    # make a list of lists that each have 10 numbers
    import random

    # Define the number of lists and the length of each list
    num_lists = 10
    list_length = 48

    # Define the range of possible elements
    min_element = 0
    max_element = 486

    # Create an empty list to store the lists
    lists = []

    # Loop through the number of lists
    for i in range(num_lists):
        # Create an empty list to store the elements
        elements = []
        # Loop through the length of each list
        for j in range(list_length):
            # Generate a random element
            element = random.randint(min_element, max_element)
            # Check if the element is already in the list
            while element in elements:
                # If yes, generate a new element
                element = random.randint(min_element, max_element)
            # Append the element to the list
            elements.append(element)
        # Append the list to the lists
        lists.append(elements)

    # Print the lists
    print(lists)
    finalList=[]
    for i in lists:
        print(f"Features {i}")
        # takeout 
        # cut the features from the training data
        X_train_cut=trainX.iloc[:,i]
        X_test_cut=testX.iloc[:,i]
        # fit the model
        Pipe=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
        Pipe.fit(X_train_cut,trainY)
        perm_importance = permutation_importance(Pipe, X_train_cut, trainY, n_repeats=1, random_state=42, n_jobs=-1)
        # Select the features with a score above a threshold
        threshold = np.quantile(perm_importance.importances_mean, 0.75) # Example threshold to select top 25% of features
        selected_features = perm_importance.importances_mean > threshold
        # Print the selected features as their column names
        print(X_train_cut.columns[selected_features])
        finalList.append(selected_features)
        i_old=i
    print("Done")
    # Select features based on importance
    print(finalList)
    
    X_train_selected = trainX[:, finalList]
    X_test_selected = testX[:, finalList]
    PipeSelected=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
    PipeSelected.fit(X_train_selected,trainY)
    Pipe=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=0.0005,C=2))])
    Pipe.fit(trainX,trainY)

    print(f"Score of Selected features:{r2_score( testY,runPredict(X_test_selected,PipeSelected))}")
    print(f"Score of All features:{r2_score( testY,runPredict(testX,Pipe))}")
def calculateMostImportantFeatures(station,time):
    data=SVRTimestampModel.load_model(station,time)
    x,y=addPreviousPowerToSplittedTrainingData(data['train_features'],data['train_power'])
    deleteCol=[]
    for col in x.columns:
        if any(x in col for x in ['windspeed','pressure','winddirection']):
                deleteCol.append(col)
    x=x.drop(deleteCol,axis=1)
    # correlation matrix of the data
    plt.figure(figsize=(16,9))
    datacollected=x
    datacollected['power']=y
    corrmatrix=datacollected.corr()
    # wind presure
    
    # take out the top 10 features that correlate with power
    # take the absolute value of the correlation
    corrmatrix=abs(corrmatrix)
    # sort the correlation matrix
    corrmatrix=corrmatrix.sort_values(by=['power'],ascending=False)
    # take the top 10 features and plot the correlation with power in a bar plot
    corrmatrix=corrmatrix[1:11]
    plt.bar(corrmatrix.index,corrmatrix['power'])
    plt.xticks(rotation=90)
    plt.ylabel("Correlation",fontsize=12)
    plt.xlabel("Feature",fontsize=12)
    plt.title("Correlation between features and power, top 10 features",fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\correlation.png",bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    calculateMostImportantFeatures(1,0)
    main()