import os, sys
from csv import writer
import matplotlib.pyplot as plt
import numpy as np


# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader
from sklearn.svm import SVR 
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
# pipeline
from sklearn.pipeline import Pipeline
import sklearn as sk
import SVRMachineLearning 
import pandas as pd
data=SVRMachineLearning.load_data_lmd_flatten(SVRMachineLearning.load_data_arrayofdays(8))
# concat the list of dataframes into one dataframe
data=data[1:-1]
data=pd.concat(data)
# drop the first 96 samples
data=data.dropna()
print("Done loading, training...")
[features,featuresTest,power,powerTest]=SVRMachineLearning.splitContinuousData(data)
scores=[]
elipson=[0.01/100,0.05/100,0.1/100,0.5/100,1/100,2/100,5/100,10/100]
C=[0.01,0.05,0.1,0.5,1,2,5,10]
parameters=elipson
best=-10000
for j in C:
    for i in parameters:
        print("New iteration")
        PipeSearch=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=i,C=j,cache_size=4000))])
        PipeSearch.fit(features,power)
        current=PipeSearch.score(featuresTest,powerTest)
        scores.append(current)
        if current>best:
            print(f"New best score: {current}")
            print(f"New best parameters: {i,j}")
            best=current
        # append to file every iteration
        with open(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Scripts\SVRGridSearch.csv",'a') as file:
            # write parameters and scores to file csv
            writer_object = writer(file)
            writer_object.writerow([i,j,current])
        
            # Close the file object
            file.close()
        print(f"Percent done: {round((len(parameters)*C.index(j)+parameters.index(i))/(len(parameters)*len(C))*100,2)}%")
            
plt.plot(parameters,scores)
plt.xlabel("Epsilon")
plt.ylabel("Score")
plt.title("Epsilon vs Score")
plt.tight_layout()
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\EpsilonVsScore.png",format="png")

#sh = HalvingGridSearchCV(PipeSearch, parameter, cv=2,
#                         -factor=2, 
#                         aggressive_elimination=False,verbose=1).fit(features, power)
#print(sh.best_params_)
#print(sh.best_estimator_.score(featuresTest,powerTest))
# plt.figure()
# plt.plot(sh.best_estimator_.predict(featuresTest))

# ModelFile="bestSVR.pkl"
# with open(ModelFile,'wb') as file:
#     pickle.dump(sh.best_estimator_,file)
    
