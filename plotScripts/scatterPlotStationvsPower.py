"Script for ploting the feature vs power for all features in the dataset"

# Importing libraries
import os
import sys
# set syspath to include the base folder
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import home made functions
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Main function
def main():
    # Importing data
    all_data = fl.loadAllPkl()
    meta_data=fl.loadFile("metadata.csv")
    powerMean = []
    globalIrrMean=[]
    labels=["Station00", "Station01", "Station02", "Station03", "Station04", "Station05","Station06","Station07","Station08","Station09"]
    for i in range(len(all_data)):
        tempDataframe = all_data[i]
        filteredPowerData=[]
        filteredGlobalIrr=[]
        for j in range(len(tempDataframe)):
            filteredPowerData = np.append(filteredPowerData,tempDataframe.iloc[j,17])
            filteredGlobalIrr = np.append(filteredGlobalIrr,tempDataframe.iloc[j,2])
        # Plotting
        powerMean = np.append(powerMean,np.mean(filteredPowerData))
        globalIrrMean = np.append(globalIrrMean,np.mean(filteredGlobalIrr))
    capacity = meta_data["Capacity"]
    #panelSize = meta_data["Panel_Size"]
    #panelNumber = meta_data["Panel_Number"]
    #totalPanelSize = panelSize*panelNumber
    plt.subplot(1, 2, 1)
    plt.scatter(capacity,powerMean)
    plt.xlim(0, max(capacity) + 5000)  # Adjust the upper limit as needed
    plt.ylim(0, max(powerMean) + 2)  # Adjust the upper limit as needed

    plt.text(capacity[0]-700, powerMean[0]-0.15, labels[0], ha='right',fontsize=12)
    plt.text(capacity[1]+700, powerMean[1]-0.15, labels[1], ha='left',fontsize=12)
    plt.text(capacity[2]-700, powerMean[2]-0.15, labels[2], ha='right',fontsize=12)
    plt.text(capacity[3]+700, powerMean[3]-0.15, labels[3], ha='left',fontsize=12)
    plt.text(capacity[4]+700, powerMean[4]-0.15, labels[4], ha='left',fontsize=12)
    plt.text(capacity[5]-700, powerMean[5]-0.15, labels[5], ha='right',fontsize=12)
    plt.text(capacity[6]-700, powerMean[6]-0.15, labels[6], ha='right',fontsize=12)
    plt.text(capacity[7]+700, powerMean[7]-0.15, labels[7], ha='left',fontsize=12)
    plt.text(capacity[8]+700, powerMean[8]-0.15, labels[8], ha='left',fontsize=12)
    plt.text(capacity[9]-700, powerMean[9]-0.15, labels[9], ha='right',fontsize=12)
    
    plt.title('Scatter Plot with mean power produced vs capacity',fontsize=16)
    plt.xlabel('Capacity [kW]',fontsize=14)
    plt.ylabel('Mean Power [MW]',fontsize=14)
    
    plt.subplot(1, 2, 2)
    plt.scatter(capacity,globalIrrMean)
    plt.xlim(0, max(capacity) + 5000)  # Adjust the upper limit as needed
    plt.ylim(0, max(globalIrrMean) + 100)  # Adjust the upper limit as needed
    plt.text(capacity[0]-700, globalIrrMean[0]-5, labels[0], ha='right',fontsize=12)
    plt.text(capacity[1]+700, globalIrrMean[1]-5, labels[1], ha='left',fontsize=12)
    plt.text(capacity[2]-700, globalIrrMean[2]-5, labels[2], ha='right',fontsize=12)
    plt.text(capacity[3]+700, globalIrrMean[3]-5, labels[3], ha='left',fontsize=12)
    plt.text(capacity[4]+700, globalIrrMean[4]-5, labels[4], ha='left',fontsize=12)
    plt.text(capacity[5]-700, globalIrrMean[5]-5, labels[5], ha='right',fontsize=12)
    plt.text(capacity[6]-700, globalIrrMean[6]-5, labels[6], ha='right',fontsize=12)
    plt.text(capacity[7]+700, globalIrrMean[7]-5, labels[7], ha='left',fontsize=12)
    plt.text(capacity[8]+700, globalIrrMean[8]-5, labels[8], ha='left',fontsize=12)
    plt.text(capacity[9]-700, globalIrrMean[9]-5, labels[9], ha='right',fontsize=12)
    plt.title('Scatter Plot with mean global irradiance vs capacity',fontsize=16)
    plt.xlabel('Capacity [kW]',fontsize=14)
    plt.ylabel('Mean Global Irradiance [W/m^2]',fontsize=14)

    
# Executing main function
if __name__ == "__main__":
    main()
    
    plt.tight_layout()
    plt.show()
