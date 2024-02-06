# Script for creating plots for report 
# import libraries
from matplotlib.font_manager import font_scalings
import numpy as np
import matplotlib.pyplot as plt
import SVRMachineLearning 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import pandas as pd
# Load data

data00=SVRMachineLearning.load_data_lmd_flatten(SVRMachineLearning.load_data_arrayofdays(0))



plt.figure(figsize=(16*0.5,9*0.5))
plt.plot(data00[0]["nwp_globalirrad"],data00[0]["nwp_temperature"],".")
plt.xlabel("Global irradiance [W/m^2]",fontsize=18)
plt.ylabel("Temperature [C]",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Example of NWP data, not scaled",fontsize=20)
plt.tight_layout()
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_not_scaled.png",format="png")


scaledData00=StandardScaler().fit_transform(data00[0].drop(columns=["date_time"]))

plt.figure(figsize=(16*0.5,9*0.5))
plt.plot(scaledData00[:,0],scaledData00[:,2],".")
plt.xlabel("Global irradiance [W/m^2]",fontsize=18)
plt.ylabel("Temperature [C]",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Example of NWP data, scaled with StandardScaler",fontsize=20)
plt.tight_layout()
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_scaled.png",format="png")
print("Nye plots")
data00=data00[1:-1]
data00=pd.concat(data00)
# drop colloumns that contain nan values
data00=data00.dropna(axis=1)
PipeSearchLinear=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='linear'))])
PipeSearchPoly=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='poly'))])
PipeSearchRadial=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf'))])

features00,featuresTest00,power00,powerTest00=SVRMachineLearning.splitContinuousData(data00)
#droppedNightFT=SVRMachineLearning.removeNightData(featuresTest00)
#droppedNightPT=SVRMachineLearning.removeNightData(pd.DataFrame(powerTest00)
droppedNightFT=featuresTest00
droppedNightPT=pd.DataFrame(powerTest00)
#convert powerTest to dataframe
PipeSearchLinear.fit(features00,power00)
PipeSearchPoly.fit(features00,power00)
PipeSearchRadial.fit(features00,power00)
# Plot the results

plotData=droppedNightFT[0:96*2]
plotPower=droppedNightPT[0:96*2]
plt.figure()
plt.plot(plotData.index,plotPower.values,label="Actual")
plt.plot(plotData.index,PipeSearchLinear.predict(plotData),label=f"Linear kernel = {PipeSearchLinear.score(droppedNightFT,droppedNightPT)}")
plt.plot(plotData.index,PipeSearchPoly.predict(plotData),label=f"Polynomial = {PipeSearchPoly.score(droppedNightFT,droppedNightPT)}")
plt.plot(plotData.index,PipeSearchRadial.predict(plotData),label=f"Radial = {PipeSearchRadial.score(droppedNightFT,droppedNightPT)}")
plt.legend(fontsize=12)
plt.xlabel("Time [UTC]",fontsize=18)
plt.ylabel("Power [MW]",fontsize=18)
plt.title("Comparison of SVR kernels",fontsize=20)  
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\SVR_kernels.png",format="png")


