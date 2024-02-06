from sklearn.linear_model import LinearRegression
import numpy as np
import LSTM as LS
from matplotlib import pyplot as plt
import fileLoader as fl

#datafile
datafile_path = 'station01'
#temp_data = fl.loadFile(datafile_path,PKL=False)

#indlæs data
cols_to_remove =['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'date_time', 'station','lmd_hmd_directirrad','nwp_hmd_diffuseirrad','nwp_pressure']
trainX,trainY,testX,testY=LS.load_LSTM_zero_padded(datafile_path,n_past=192,n_future=192)

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

print(1)

"""
for i in range(1,97):
    X_train_slice1 = trainX[:, 0:1, :]
    X_train_slice2 = trainX[:, i:i + 1, :]
    X_train_slice = np.concatenate((X_train_slice1, X_train_slice2), axis=1)
    y_train_slice = trainY[:, i-1, 0]
    
    X_test_slice1 = testX[:,0:1,:]
    X_test_slice2 = testX[:,i:i+1,:]
    X_test_slice = np.concatenate((X_test_slice1, X_test_slice2), axis=1)
    y_test_slice = testY[:,i-1,0]  
    
    # Append slices to lists
    X_train_list.append(X_train_slice)
    y_train_list.append(y_train_slice)
    
    X_test_list.append(X_test_slice)
    y_test_list.append(y_test_slice)

"""
# Create a linear regression model
models = []

for i in range(192):
    model = LinearRegression()
    models.append(model)
    print(i)

# Fit the model to your data
reshaped_data = np.reshape(trainX, (26420, 1, 1920))
reshaped_data_to_fit = reshaped_data[:,0,:]
for i in range(192):
    #models[i].fit(reshaped_data_to_fit, y_train_list[i])
    models[i].fit(reshaped_data_to_fit, trainY[:,i,0])
    print(i)

print(2)
# Get the optimal weights
#weights = model.coef_
#bias = model.intercept_

#print("Optimal Weights:", weights)
#print("Bias:", bias)
y_pred=[]
reshaped_data = np.reshape(testX, (6605, 1, 1920))
reshaped_data_to_predict = reshaped_data[:,0,:]
for i in range(192):
    predicted_temp = models[i].predict(reshaped_data_to_predict)
    y_pred.append(predicted_temp)

# 'y_pred' now contains the predicted output for the test set

# Evaluate the performance on the test set (you may use different metrics based on your problem)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Assuming y_test is the true output for the test set
# Calculate Mean Squared Error (MSE) as an example
trainX_zero,trainY_zero,testX_zero,testY_zero = LS.load_LSTM_zero_padded('station01', n_past=192, n_future=192)

print(3)
zero_indices=[]

for i in range(192):
    temp = testY_zero[:,i,0]
    # Create a mask for zero values
    zero_mask = (temp == 0)

    # Get the indices where the values are zero
    zero_indices.append(np.where(zero_mask)[0])

y_test_list_day=[]
y_pred_day = []

print(4)
for i in range(192):
    # Use fancy indexing to remove rows at zero_indices
    y_test_list_day_temp = np.delete(testY[:,i,0], zero_indices[i], axis=0)
    y_test_list_day.append(y_test_list_day_temp)
    
    y_pred_day_temp = np.delete(y_pred[i][:], zero_indices[i], axis=0)
    y_pred_day.append(y_pred_day_temp)


print(5)  
         
        
        

mse=[]
R_2=[]
for i in range(192):
    mse_temp = mean_squared_error(y_test_list_day[i], y_pred_day[i])
    R_2_temp =r2_score(y_test_list_day[i], y_pred_day[i])
    mse.append(mse_temp)
    R_2.append(R_2_temp)
    
mse_array = np.array(mse)
R_2_array = np.array(R_2)

time_steps_hours = np.arange(0, 24, 0.25)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Plot MSE
ax1.plot(time_steps_hours, mse_array, marker='o', linestyle='-', color='blue')
ax1.set_title('Mean Squared Error (MSE)')
ax1.set_xlabel('Time into the future (hours)')
ax1.set_ylabel('MSE')

# Plot R-squared
ax2.plot(time_steps_hours, R_2_array, marker='o', linestyle='-', color='green')
ax2.set_title('R-squared')
ax2.set_xlabel('Time into the future (hours)')
ax2.set_ylabel('R-squared')

# Adjust layout for better readability
plt.tight_layout()

# Show the plots
plt.show()
