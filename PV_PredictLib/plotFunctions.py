import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from PV_PredictLib import fileLoader as fl



def plotBase(ax,x,y,label):
    ax.plot(x,y,".",label=label)
    ax.legend()
    
def plotTimeSeries(ax :plt.axes ,data : pd.DataFrame,colloumName : str,label: str ,scaleTicks : float= 2):
    x = data["date_time"]
    y = data[colloumName]
    plotBase(ax,x,y,label)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(colloumName)
    #set ticks 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=scaleTicks))
    ax.xaxis.set_tick_params(rotation=90)
    #set ticks to be 90 rotated
    #ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    return ax    

def plotColumnScatter(ax :plt.axes ,data : pd.DataFrame,colloum1Name : str,colloum2Name : str,label: str):
    x = data[colloum1Name]
    y = data[colloum2Name]
    plotBase(ax,x,y,label)
    ax.set_xlabel(colloum1Name)
    ax.set_ylabel(colloum2Name)
    #set legend location to upper right
    ax.legend(loc='upper right',fontsize=12)
    return ax
# plot columnScatter with two y axis's
def plotColumnScatter2Y(ax :plt.axes ,data : pd.DataFrame,colloumXName : str,colloumY1Name : str,colloumY2Name : str,label: str ,scaleTicks : float= 2):
    x = data[colloumXName]
    y1 = data[colloumY1Name]
    y2 = data[colloumY2Name]
    ax.spines['right'].set_color(c='C0')
    ax.tick_params(axis='y', colors='C0')
    ax.set_ylabel(colloumY2Name,color='C0')
    ax.plot(x,y1,".",label=label,color='C0')
    ax.set_xlabel(colloumXName)
    ax.set_ylabel(colloumY1Name)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=scaleTicks))
    ax.xaxis.set_tick_params(rotation=90)
    ax2 = ax.twinx()
    # ensure another color for the second y axis
    ax2.spines['right'].set_color('C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.set_ylabel(colloumY2Name,color='C1')
    ax2.plot(x,y2,".",label=label,color='C1')
    ax2.set_ylabel(colloumY2Name)
    return ax

def plotHistogram(ax :plt.axes ,data : pd.DataFrame,colloumName : str,label: str,binCount=20):
    x = data[colloumName]
    #normalize density for better histogram
    x.plot.hist(ax=ax,label=label+" histogram",density=True,bins=binCount)
    x.plot.kde(ax=ax,legend=True,label=label+" kde")
    ax.set_xlabel(colloumName)
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax

def correlationMatrixPlotter(ax :plt.axes ,data : pd.DataFrame):  
    #drop non float colloums
    data2 = data.select_dtypes(include=['float64'])
    ax = sns.heatmap(data2.corr(), ax=ax,annot=True, fmt=".1f")
    return ax    

def plot_means_and_variances(stats):
    """
    Plot the means and variances for each corresponding column.
    
    Parameters:
        stats (pd.DataFrame): DataFrame containing 'DataFrame', 'Column', 'Average', and 'Variance'.
    """
    unique_columns = stats['Column'].unique()

    for col in unique_columns:
        col_stats = stats[stats['Column'] == col]
        data_frames = col_stats['DataFrame']
        averages = col_stats['Average']
        variances = col_stats['Variance']

        plt.figure(figsize=(10, 5))
        
        # Scatter plot with DataFrame numbers as legends
        for i, (variance, average, dataframe) in enumerate(zip(variances, averages, data_frames), 1):
            plt.scatter(variance, average, label=f'DF {dataframe}', c=f'C{i}', cmap='viridis')
        
        plt.xlabel('Variance')
        plt.ylabel('Mean []')
        plt.title(f'Means vs. Variances for Column {col}')
        plt.legend(title='DataFrame Number')
        plt.show()

def powerHeatMap(ax :plt.axes ,data : pd.DataFrame):
    data2 = data.select_dtypes(include=['float64'])
    print(data2.corr())

def plotPowCorr(data):
    """
    This function plots a heatmap of the correlation between power and NWP data for each power station
    """
    correlation = np.zeros(13)
    vectors = []
    for i in range(len(data)):
        temp = data[i]
        #temp = temp.select_dtypes(include=['float64'])
        correlation[0] = temp["power"].corr(temp["lmd_diffuseirrad"])
        correlation[1] = temp["power"].corr(temp["lmd_hmd_directirrad"])
        correlation[2] = temp["power"].corr(temp["lmd_totalirrad"])
        correlation[3] = temp["power"].corr(temp["lmd_pressure"])
        correlation[4] = temp["power"].corr(temp["lmd_temperature"])
        correlation[5] = temp["power"].corr(temp["lmd_windspeed"])
        correlation[6] = temp["power"].corr(temp["nwp_hmd_diffuseirrad"])
        correlation[7] = temp["power"].corr(temp["nwp_directirrad"])
        correlation[8] = temp["power"].corr(temp["nwp_globalirrad"])
        correlation[9] = temp["power"].corr(temp["nwp_humidity"])
        correlation[10] = temp["power"].corr(temp["nwp_pressure"])
        correlation[11] = temp["power"].corr(temp["nwp_temperature"])
        correlation[12] = temp["power"].corr(temp["nwp_windspeed"])        
        vectors.append(correlation)
        correlation = np.zeros(13)
    powCorrMatrix = np.array(vectors)
    # labels for x-axis
    x_axis_labels = ["LMD Diffuseirrad","LMD HMD Directirrad","LMD Totalirrad","LMD Pressure","LMD Temperature","LMD Windspeed", "NWP HMD Diffuseirrad", "NWP Directirrad", "NWP Globalirrad", "NWP Humidity", "NWP Pressure", "NWP Temperature","NWP Windspeed"] 
    # labels for y-axis
    y_axis_labels = ["Station00","Station01","Station02","Station03","Station04","Station05","Station06","Station07","Station08","Station09"] 
    powCorrMatrix = pd.DataFrame(powCorrMatrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax = sns.heatmap(powCorrMatrix, ax=ax,vmin = -1, vmax = 1,linewidths=.5,annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt=".2f", cbar=False)
    ax.set_title("Correlation matrix of power and each recorded feature from the 10 stations", fontsize=16)
    plt.tight_layout()
  
def circle3dScatterPlot(dataFrame,setting,namestring):
    """
    This functions makes a scatter plot in 3d of 
    the wind drection converted intro sin an cos values
    It can either be set to plot the "average" of 
    measured power in 1 degrees intervals or plot the
    "individual" points of data
    """
    #name  = globals()[dataFrame]
    if setting=="average":
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,1,1,projection='3d')


        nwp_winddirection = dataFrame["nwp_winddirection"].to_numpy()
        power = dataFrame.iloc[:, 14].to_numpy()

        # Initialize arrays to store sums and counts for each angle
        gns_power = np.zeros(360, dtype=float)
        power_indeks = np.zeros(360, dtype=int)

        for angle in range(360):
            angle_range = (angle <= nwp_winddirection) & (nwp_winddirection <= angle + 1)

            # Calculate sums and counts for the current angle
            gns_power[angle] = np.sum(power[angle_range])
            power_indeks[angle] = np.sum(angle_range)

        # Avoid division by zero and compute the final average
        power_indeks_nonzero = power_indeks > 0
        gns_power[power_indeks_nonzero] /= power_indeks[power_indeks_nonzero]


        x = np.array(list(range(360)) )

        W1 = [np.cos(x*np.pi/180), np.sin(x*np.pi/180)]
        ax.scatter(W1[0],W1[1],gns_power)

        ax.set_xlabel('Cosinus')
        ax.set_ylabel('Sinus')
        ax.set_zlabel('average power [MW]')
        ax.set_title(namestring + ' average power compared to wind direction')
        
    elif setting=="individual":
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        windDirConvert = [np.cos(nwp_winddirection*np.pi/180), np.sin(nwp_winddirection*np.pi/180)]


        ax.scatter(windDirConvert[0],windDirConvert[1],power)

        ax.set_xlabel('Cosinus')
        ax.set_ylabel('Sinus')
        ax.set_zlabel('power [MW]')
        ax.set_title(namestring + ' average powewr compared to wind direction')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        

def nwpError():
    dataTemp = fl.loadAllPkl()
    for i in range(10):
        temp = dataTemp[i]
        if i == 0:
            data = temp
        else:
            pd.concat([data, temp])
#    data=fl.load_all_datasets()

    MSE_windspeed=mean_squared_error(data.lmd_windspeed,data.nwp_windspeed)
    RMSE_windspeed=np.sqrt(MSE_windspeed)
    NRMSE_windspeed=RMSE_windspeed/np.mean(data.lmd_windspeed)
    print('windspeed NRMSE: ',NRMSE_windspeed)

    MSE_pressure=mean_squared_error(data.lmd_pressure,data.nwp_pressure)
    RMSE_pressure=np.sqrt(MSE_pressure)
    NRMSE_pressure=RMSE_pressure/np.mean(data.lmd_pressure)
    print('pressure NRMSE: ',NRMSE_pressure)

    MSE_temperature=mean_squared_error(data.lmd_temperature,data.nwp_temperature)
    RMSE_temperature=np.sqrt(MSE_temperature)
    NRMSE_temperature=RMSE_temperature/np.mean(data.lmd_temperature)
    print('temperature NRMSE: ',NRMSE_temperature)

    MSE_globalirrad=mean_squared_error(data.lmd_totalirrad,data.nwp_globalirrad)
    RMSE_globalirrad=np.sqrt(MSE_globalirrad)
    NRMSE_globalirrad=RMSE_globalirrad/np.mean(data.lmd_totalirrad)
    print('globalirrad NRMSE: ',NRMSE_globalirrad)
    
    MSE_directirrad=mean_squared_error(data.lmd_hmd_directirrad,data.nwp_directirrad)
    RMSE_directirrad=np.sqrt(MSE_directirrad)
    NRMSE_directirrad=RMSE_directirrad/np.mean(data.lmd_hmd_directirrad)
    print('directlirrad NRMSE: ',NRMSE_directirrad)
    
    MSE_diffuseirrad=mean_squared_error(data.lmd_diffuseirrad,data.nwp_hmd_diffuseirrad)
    RMSE_diffuseirrad=np.sqrt(MSE_diffuseirrad)
    NRMSE_diffuseirrad=RMSE_diffuseirrad/np.mean(data.lmd_diffuseirrad)
    print('diffuseirrad NRMSE: ',NRMSE_diffuseirrad)


    labels = ['Temperature', 'Pressure', 'Wind speed', 'Global Irradiance', 'Direct Irradiance', 'Diffuse Irradiance']  # Updated labels
    nrmse_values = [NRMSE_temperature, NRMSE_pressure, NRMSE_windspeed, NRMSE_globalirrad,NRMSE_directirrad,NRMSE_diffuseirrad]  # Updated NRMSE values

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, nrmse_values)  # Added color for Global Irradiance

    for bar, nrmse_value in zip(bars, nrmse_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(nrmse_value, 3), ha='center', va='bottom', color='black', fontsize=10)

    plt.xlabel('Variables')
    plt.ylabel('NRMSE')
    plt.title('NRMSE for Different Variables')
    plt.show()

    
    return

def windDirectionCorrelation(data):
    # Extract columns
    power = data['power']
    wind_direction_lmd = data['lmd_winddirection']
    wind_direction_nwp = data['nwp_winddirection']

    # Convert wind_direction to radians
    wind_direction_rad_lmd = np.radians(wind_direction_lmd)
    wind_direction_rad_nwp = np.radians(wind_direction_nwp)

    # Create matrix A
    A_lmd = np.vstack((np.sin(wind_direction_rad_lmd), np.cos(wind_direction_rad_lmd))).T
    A_nwp = np.vstack((np.sin(wind_direction_rad_nwp), np.cos(wind_direction_rad_nwp))).T

    # Calculate conditioning number of A
    conditioning_number_lmd = np.linalg.cond(A_lmd)
    conditioning_number_nwp = np.linalg.cond(A_nwp)

    # Calculate coefficients
    x_n_lmd = np.linalg.inv(A_lmd.T @ A_lmd) @ A_lmd.T @ power
    x_n_nwp = np.linalg.inv(A_nwp.T @ A_nwp) @ A_nwp.T @ power

    # Calculate regression
    Reg_lmd = A_lmd @ x_n_lmd
    Reg_nwp = A_nwp @ x_n_nwp

    # Calculate correlation coefficient
    correlation_lmd = np.corrcoef(Reg_lmd, power)[0, 1]
    correlation_nwp = np.corrcoef(Reg_nwp, power)[0, 1]

    print("Conditioning number lmd:", conditioning_number_lmd,'; nwp:',conditioning_number_nwp)
    print("Coefficients (a1, a2) for lmd:", x_n_lmd,'; for nwp:',x_n_nwp)
    print("Correlation coefficient lmd:", correlation_lmd,'; for nwp:',correlation_nwp)
    return correlation_lmd,correlation_nwp

def plotAvgPowerVsCap():
    fig, ax = plt.subplots()
    mean=[]
    capacity=[]
    stations = ['Station 00', 'Station 01', 'Station 02', 'Station 03', 'Station 04','Station 05','Station 06','Station 07','Station 08','Station 09']
    
    meta=fl.loadFile(f"metadata.csv")
    
    data = fl.loadAllPkl()
    for i in range(10):
        tempData= data[i]
        tempMean = tempData["power"].max()
        mean.append(tempMean)
        tempMeta = meta["Capacity"][i]
        capacity.append(tempMeta)
    

    print(capacity)
    print(mean)
    ax.scatter(capacity,mean)
    
    
    ax.annotate(stations[0], (capacity[0], mean[0]), textcoords="offset points", xytext=(10,00), ha='left')
    ax.annotate(stations[1], (capacity[1], mean[1]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(stations[2], (capacity[2], mean[2]), textcoords="offset points", xytext=(0,10), ha='center')
        
    ax.annotate(stations[3], (capacity[3], mean[3]), textcoords="offset points", xytext=(10,-2), ha='left')
    ax.annotate(stations[4], (capacity[4], mean[4]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(stations[5], (capacity[5], mean[5]), textcoords="offset points", xytext=(-10,0), ha='right')
    ax.annotate(stations[6], (capacity[6], mean[6]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(stations[7], (capacity[7], mean[7]), textcoords="offset points", xytext=(10,-9), ha='left')
    ax.annotate(stations[8], (capacity[8], mean[8]), textcoords="offset points", xytext=(10,3), ha='left')
    ax.annotate(stations[9], (capacity[9], mean[9]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel('Capacity [kW]')
    plt.ylabel('Mean [MW]')
    plt.title('Scatter Plot with mean power produced vs capacity')
    return  



