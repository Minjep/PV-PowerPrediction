from PV_PredictLib import fileLoader as fl
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(st_data, string, pressure):

    if string == "lmd":
        st_data = st_data[st_data.columns[0:7]]
        # remove pressure as it is very highly correlated with everything else,
        # i.e. it can be described by the other features very well
        if not pressure:
            st_data = st_data.drop("lmd_pressure", axis=1)
    else:
        st_data = st_data[st_data.columns[7:15]]
        if not pressure:
            st_data = st_data.drop("nwp_pressure", axis=1)

    # printing first few rows
    print(st_data.head())

    # the independent variables set
    X = st_data[st_data.columns]

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    # So what his function essentially does is it takes the ith column and calculates the correlation
    # with the linear combination of the rest of the features' columns.
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]

    print(vif_data)

    # Make a bar plot of the data
    feature = vif_data['feature'].head(12)
    vif = vif_data['VIF'].head(12)

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))
    ax_left = 0.185
    ax_bottom = 0.11
    ax_width = 0.9-ax_left
    ax_height = 0.88-ax_bottom
    ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

    # Horizontal Bar Plot
    ax.barh(feature, vif, color='g')

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel('VIF', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=24, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title(string.upper() + ' features and their corresponding VIF',
                 loc='left', fontsize=24)

    if not pressure:
        plt.savefig(string + "_vif_no_pressure.png")
    else:
        plt.savefig(string + "_vif_with_pressure.png")

    # Show Plot
    plt.show()


def main():
    for i in range(10):
        print("Loading station", i)
        st_data = fl.loadPkl(f"station0{i}.pkl")

    # drop non-numeric cols
    st_data = st_data._get_numeric_data()

    # remove station number as it is not a feature
    st_data = st_data.drop("station", axis=1)
    # power is not a feature either, it is what we want to predict using features
    st_data = st_data.drop("power", axis=1)

    # Calculate the vif factor
    vif(st_data, string="nwp", pressure=True)

if __name__ == "__main__":
    main()