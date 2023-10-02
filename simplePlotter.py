#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf

#Start by listing data files in PVO dataset
station02 = fileLoader.loadFile("station02.csv")
fileLoader.fileInfo(station02)

# EXAMPLE Andreas PC
# try:
#     station02 = fileLoader.loadFile("station02.csv", r'C:\Users\andre\OneDrive - Aalborg Universitet\_Universitet\ES7\_ES7 project\literatureAndDataset\dataset')
# except:
#     pass
# # EXAMPLE Jeppe PC
# try:
#     station02 = fileLoader.loadFile("station02.csv", r'C:\Users\jeppe\gits\PV-PowerPrediction\dataset')
# except:
#     pass

#Check data
date_fails = np.zeros(10)
out_and_empty_fails=np.zeros((10, 2))
for i in range(10):
    date_fails[i] = fileLoader.checkDate(stations[i])
    out_and_empty_fails[i]=fileLoader.checkParam(stations[i],8)
print(date_fails)
print(np.shape(out_and_empty_fails))
print(out_and_empty_fails)

# print(station02.head())
station02_sliced= fileLoader.sliceData(station02,"2018-07-22 16:00:00","2018-07-22 19:00:00")
# Create a figure to plot on, 2 axes in 1 column
[fig,ax]=plt.subplots(2,2,figsize=(10,10))
# access the first axis by ax[0] and the second by ax[1]
pf.plotTimeSeries(ax[0][0],station02_sliced,"power","power")
pf.plotColumnScatter2Y(ax[0][1],station02,"power","lmd_windspeed","nwp_temperature","power vs windspeed")
pf.plotHistogram(ax[1][0],station02,"power","power")
plt.tight_layout()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
pf.correlationMatrixPlotter(ax2,station02)
ax2.set_title("Correlation matrix of dataset")

plt.tight_layout()


plt.show()

