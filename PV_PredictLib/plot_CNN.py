import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import os, sys
import numpy as np

def main():
    file_path = "../../Implementation/CNN_test3_1output/UNzip"
    #file_path = "/media/diana-valeria/Expansion/Valeria/Documents/AAU/Semester7/P7/Implementation/CNN_test3_1output/UNzip"

    data = []
    # if os.path.isfile(file_path):
    #     print(f'The file {file_path} exists ')
    for i in range(96):
        with open(file_path + "/CNN_test" + str(i) + ".pkl", 'rb') as file:
            data_i = pickle.load(file)
            data.append(data_i)
            # power_predictions = pickle.load(file)
            # power_pred_day = pickle.load(file)
            # testY = pickle.load(file)
            # testY_day = pickle.load(file)
            # r2_day = pickle.load(file)
            # mse_day = pickle.load(file)

    r2_list = []
    mse_list = []
    pred_list = []
    test = []
    one_day_pred = []
    one_day_true = []
    for i in range(96):
        r2_list.append(data[i]['r2'])
        mse_list.append(data[i]['mse'])
        pred_list.append(data[i]['predictions_day'])
        test.append(data[i]['testY_day'])
        one_day_pred.append(data[i]['predictions'][95])
        one_day_true.append(data[i]['testY'][i, 95])
        # one_day_true.append(data[i]['testY_day'][95])

    time_steps_hours = np.arange(0, 24, 0.25)

    plt.figure()
    plt.plot(time_steps_hours, r2_list, marker='.')
    plt.title('$R^2$ score for day data', fontsize=20)
    plt.xlabel('Hours', fontsize=15)
    plt.ylabel('$R^2$', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('r2_day_score.png')
    plt.show()

    plt.figure()
    plt.plot(time_steps_hours, mse_list, marker='.')
    plt.title('Mean Squared Error for day data', fontsize=20)
    plt.xlabel('Hours', fontsize=15)
    plt.ylabel('MSE', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('mse_day_score.png')
    plt.show()

    k = 95
    plt.figure()
    #plt.plot(data[k]['testY'][:, k], label='true value')
    plt.plot(data[k]['testY_day'], label='true value')
    plt.plot(data[k]['predictions_day'], label='predicted')
    plt.legend()
    plt.title('Actual and predicted power for station 01', fontsize=15)
    plt.ylabel('Power [MW]', fontsize=15)
    plt.xlabel('Time', fontsize=15)
    plt.savefig('Pred_vs_true.png')
    plt.show()

    plt.figure()
    plt.plot(one_day_pred, label='predicted')
    plt.plot(one_day_true, label='true value')
    plt.legend()
    plt.title('Actual and predicted power for station 01 for one day', fontsize=15)
    plt.ylabel('Power [MW]', fontsize=15)
    plt.xlabel('Time', fontsize=15)
    plt.savefig('Pred_vs_true_one_day.png')
    plt.show()

    test = 0


if __name__ == '__main__':
    main()