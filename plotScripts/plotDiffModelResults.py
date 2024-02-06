"Script for ploting the correlation matrix of the station02 dataset"

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
import seaborn as sb 
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def main():
    # Importing results
    file_path = "/Users/jakob/Desktop/JakobTemp/JakobTemp1Layers/gridSearchResults1.pkl"
    temp = open(file_path, 'rb')
    df1 = pickle.load(temp)
    temp.close()
    file_path = "/Users/jakob/Desktop/JakobTemp/JakobTemp2Layers/gridSearchResults2.pkl"
    temp = open(file_path, 'rb')
    df2 = pickle.load(temp)
    temp.close()
    file_path = "/Users/jakob/Desktop/JakobTemp/JakobTemp3Layers/gridSearchResults3.pkl"
    temp = open(file_path, 'rb')
    df3 = pickle.load(temp)
    temp.close()
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    combined_df = combined_df.sort_values(by='MSE')
    file_path = "/Users/jakob/Desktop/gridSearchResults100/gridSearchResults400.pkl"
    temp = open(file_path, 'rb')
    df4 = pickle.load(temp)
    temp.close()        
    
    
    combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    combined_df = combined_df.sort_values(by='MSE')
    
    for i in range(combined_df.shape[0]):
        print(combined_df.iloc[i])
    
    
    
    
    results_df = combined_df[combined_df['MSE'] <= 0.1]
    
    
    # Extract the columns
    num_layers = results_df['Num Layers']
    num_neurons = results_df['Num Neurons']
    #batch_size = results_df['Batch Size']
    #val_split = results_df['Validation Split']
    
    mse = results_df['MSE']
    #R_squared = results_df['R^2']
    
    df_sorted = results_df.sort_values(by='MSE')
    for i in range(len(df_sorted)):
        print(df_sorted[i])
    mse_dict = {}

    # Loop through unique values and create vectors
    layer1Mean = np.mean(results_df.loc[results_df['Num Layers'] == 1, 'MSE'].values)
    layer2Mean = np.mean(results_df.loc[results_df['Num Layers'] == 2, 'MSE'].values)
    layer3Mean = np.mean(results_df.loc[results_df['Num Layers'] == 3, 'MSE'].values)
    
    neuron100Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 100, 'MSE'].values)
    neuron200Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 200, 'MSE'].values)
    neuron400Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 400, 'MSE'].values)
    neuron800Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 800, 'MSE'].values)
    
    batch4Mean = np.mean(results_df.loc[results_df['Batch Size'] == 4, 'MSE'].values)
    batch8Mean = np.mean(results_df.loc[results_df['Batch Size'] == 8, 'MSE'].values)
    batch16Mean = np.mean(results_df.loc[results_df['Batch Size'] == 16, 'MSE'].values)
    
    val00Mean = np.mean(results_df.loc[results_df['Validation Split'] == 0.0, 'MSE'].values)
    val01Mean = np.mean(results_df.loc[results_df['Validation Split'] == 0.1, 'MSE'].values)
    

    # Create a scatter plot
    plt.figure(1)
    plt.scatter(num_layers, mse, marker='o', color='blue')
    plt.scatter(1, layer1Mean, marker='o',s=100 , color='red', label='Layer 1 Mean')
    plt.scatter(2, layer2Mean, marker='o',s=100, color='red', label='Layer 2 Mean')
    plt.scatter(3, layer3Mean, marker='o',s=100, color='red', label='Layer 3 Mean')

    # Set labels and title
    plt.xlabel('Number of Layers')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Scatter Plot of Number of Layers vs MSE')
    
    plt.figure(2)
    plt.scatter(num_neurons, mse, marker='o', color='blue')
    plt.scatter(100, neuron100Mean, marker='o',s=100 , color='red', label='Neuron 100 Mean')
    plt.scatter(200, neuron200Mean, marker='o',s=100, color='red', label='Neuron 200 Mean')
    plt.scatter(400, neuron400Mean, marker='o',s=100, color='red', label='Neuron 400 Mean')
    plt.scatter(800, neuron800Mean, marker='o',s=100, color='red', label='Neuron 800 Mean')

    # Set labels and title
    plt.xlabel('Number of Neurons')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Scatter Plot of Number of neurons vs MSE')
    
    plt.figure(3)
    plt.scatter(batch_size, mse, marker='o', color='blue')
    plt.scatter(4, batch4Mean, marker='o',s=100 , color='red', label='Batch Size 04 Mean')
    plt.scatter(8, batch8Mean, marker='o',s=100, color='red', label='Batch Size 08 Mean')
    plt.scatter(16, batch16Mean, marker='o',s=100, color='red', label='Batch Size 16 Mean')

    # Set labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Scatter Plot of Batch Size vs MSE')
    
    plt.figure(4)
    plt.scatter(val_split, mse, marker='o', color='blue')
    plt.scatter(0, val00Mean, marker='o',s=100 , color='red', label='Validation Split 0.0 Mean')
    plt.scatter(0.1, val01Mean, marker='o',s=100, color='red', label='Validation Split 0.1 Mean')

    # Set labels and title
    plt.xlabel('Validation Split')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Scatter Plot of Validation Split vs MSE')


    
if __name__ == "__main__":
    main()
    plt.show()