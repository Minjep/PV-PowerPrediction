"Script for ploting the feature vs power for all features in the dataset"

# Importing libraries
import os
import sys

from six import b
# set syspath to include the base folder
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import home made functions
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf
import pandas as pd
import matplotlib.pyplot as plt

# Defining functions 

def plot_feature_vs_power(all_data,save_path,feature_name,label=""):
    fig,ax=plt.subplots(1,1,figsize=(16*0.75,9*0.75))
    # cut out unit from feature name
    unit=feature_name.split("[")[1].split("]")[0]
    feature_name=feature_name.split("[")[0]
    pf.plotColumnScatter(ax,all_data,feature_name,"power",f"{feature_name} vs power")
    ax.set_xlabel(f"{feature_name} [{unit}]",fontsize=14)
    ax.set_ylabel(" Power [MW]",fontsize=14)
    if label=="Normalized_":
        ax.set_ylabel("Normalized power",fontsize=14)
        
    plt.title(f"Scatterplot of {label}{feature_name} vs power",size=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if not os.path.exists(save_path+"/feature_vs_power"):
        os.makedirs(save_path+"/feature_vs_power")
    plt.tight_layout()
    plt.savefig(f"{save_path}/feature_vs_power/{label}{feature_name}_vs_power.png",format="png",bbox_inches='tight')
# Main function
def main():
    # Importing data
    all_data = fl.load_all_datasets()
    norm_all_data = fl.load_all_datasets(norm=True)
    print(all_data.head())
    all_data=all_data.resample('1H').first()
    norm_all_data=norm_all_data.resample('1H').first()
    # nwp features
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)    
    
    features = [r"nwp_globalirrad[$W/m^2$]", r"nwp_directirrad[$W/m^2$]",r"nwp_temperature[C°]", r"nwp_humidity[%]", r"nwp_windspeed[$m/s$]", r"nwp_pressure[hPa]"]
    # Plotting
    for feature in features:
        pathForFigures=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\Appendix DatasetFeatures"
        plot_feature_vs_power(all_data,pathForFigures,feature)
        plot_feature_vs_power(norm_all_data,pathForFigures,feature,"Normalized_")
    test=fl.sliceData(all_data,"2018-10-21 00:00:00","2018-10-25 23:59:59")
    #plt.figure()
    #plt.plot(test["power"],".")
    #plt.show()
    
# Executing main function
if __name__ == "__main__":
    main()
    data_single_station=fl.loadPkl("station05.pkl")
    data_single_station=data_single_station.resample('1H').first()
    pathForFigures=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\Appendix DatasetFeatures"
    plot_feature_vs_power(data_single_station,pathForFigures,"nwp_temperature[C°]")
    plt.show()
