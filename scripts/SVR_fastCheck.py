import matplotlib.pyplot as plt
from matplotlib import cm, markers
import numpy as np
import pandas as pd


sharedFolder=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project"
# import csv file fromm grid search
data=pd.read_csv(sharedFolder+r"\Scripts\SVRGridSearch.csv")
data.columns=["Epsilon","C","Score"]
xraw=data["Epsilon"][2:].to_numpy()
yraw=data["C"][2:].to_numpy()
zraw=data["Score"][2:].to_numpy()
# reshape the data to 2d
x=xraw.reshape(8,8)
y=yraw.reshape(8,8)
z=zraw.reshape(8,8)
# 3D surface plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,10))
cmap=plt.get_cmap("viridis")
cmap.set_under("black")
ax.plot_surface(x, y, z, vmin=0.78,vmax=0.822 ,cmap=cmap)

# 3D scatter plot 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,9))
ax.scatter(x, y, z, vmin=0.78,vmax=0.822,cmap=cmap)
ax.set_xlabel("Epsilon")
ax.set_ylabel("C")
ax.set_zlabel("R2 Score")
plt.tight_layout()
plt.suptitle("Performance surface of epsilon and C",fontsize=20)
plt.savefig(sharedFolder+r"\Figures\PerformanceSurface.png",format="png",bbox_inches="tight", pad_inches=0.2)

fig,ax=plt.subplots(2,1,figsize=(16/1.5,9/1.5))
l1=ax[0].plot(x[1],z[1],".-",color="C0",label=f"C={y[1][0]}")
ax[0].plot(x[2],z[2],".-",color="C1",label=f"C={y[2][0]}")
ax[1].plot(x[3],z[3],".-",color="C2",label=f"C={y[3][0]}")
ax[1].plot(x[4],z[4],".-",color="C3",label=f"C={y[4][0]}")
ax[1].plot(x[5],z[5],".-",color="C4",label=f"C={y[5][0]}")
ax[1].plot(x[6],z[6],".-",color="C5",label=f"C={y[6][0]}")    
ax[1].plot(x[7],z[7],".-",color="C6",label=f"C={y[7][0]}")
ax[1].scatter(x[5,1],z[5,1],marker="x",sizes=[50],color="C4",label=f"Max R2 score e={x[5,1]} C={y[5,1]}")

ax[0].set_xlabel("Epsilon")
ax[0].set_ylabel("R2 Score")
ax[1].set_xlabel("Epsilon")
ax[1].set_ylabel("R2 Score")
ax[0].set_title("C=0.05-0.1")
ax[1].set_title("C=0.5-10")
plt.suptitle("Epsilon vs R2 Score",fontsize=20)

ax[1].set_ylim(0.78,0.83)
ax[0].grid()
ax[1].grid()
ax[0].set_xscale("log")
ax[1].set_xscale("log")

plt.tight_layout()
fig.legend(bbox_to_anchor=(1, 0.6),title="Legends", title_fontsize=14,fancybox=True)
plt.subplots_adjust(right=0.77)
plt.savefig(sharedFolder+r"\Figures\EpsilonVsScore.png",format="png",bbox_inches='tight')
