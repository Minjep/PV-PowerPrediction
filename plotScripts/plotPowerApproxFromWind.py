# Simple script to plot wind direction correlation 
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import libraries
import matplotlib.pyplot as plt
from PV_PredictLib import fileLoader
from PV_PredictLib import plotFunctions as pf

station00=fileLoader.loadFile('station00.csv')

Reg_lmd,Reg_nwp=pf.windDirectionPowerApprox(station00)

n=500

Reg_lmd=Reg_lmd[:n]
Reg_nwp=Reg_nwp[:n]
data=station00['power'].values
data=data[:n]

# Plotting the variables
plt.plot(Reg_lmd, label='Reg_lmd')
plt.plot(Reg_nwp, label='Reg_nwp')
plt.plot(data, label='Measured power')

# Adding labels and title
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Plot of Reg_lmd, Reg_nwp, and Data')

# Adding a legend
plt.legend()

# Display the plot
plt.show()