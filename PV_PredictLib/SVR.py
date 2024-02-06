from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from PV_PredictLib import fileLoader  # external library for loading data files.
from datetime import datetime

from dateutil import parser
from timezonefinder import TimezoneFinder
import pytz
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))   
from sunsetSunriseAPI import sunrisesunset  

class SVRMachineLearning:
    """
    A class that encapsulates the support vector regression model for predicting
    solar power generation based on historical and meteorological data.
    """
    
    def __init__(self, pv_station_number: int):    
        """
        Initializes the SVR model pipeline and variables. It standardizes the data
        and applies SVM with a radial basis function kernel.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('SVR', SVR(kernel='rbf',epsilon=0.0005, C=2))])
        self.data_raw = fileLoader.loadFile(f"station0{pv_station_number}.csv", path=None, PKL=False)
        self.meta = fileLoader.loadFile("metadata.csv")
        self.longitude = 0
        self.latitude = 0
        self.pv_station_number = pv_station_number
    def load_days_from_timeoffset(self, hour_of_the_day: int):
        """
        Loads data per day grouped around a specified hour considering timezone offset.

        :param pv_station_number: The number of the PV station from which to load data.
        :param hour_of_the_day: The reference hour around which the data is grouped.
        :return: A list of sub-dataframes, each corresponding to one day's data.
        """
        # Load station data and metadata.
        
        data_=self.data_raw
        # Drop unwanted columns.
        self.data = data_.drop(columns=['nwp_pressure', 'lmd_pressure'])
        
        # Convert time column to datetime and group by date.
        time_column = "date_time"
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        meta_station = self.meta[self.meta["Station_ID"] == f"station0{self.pv_station_number}"]
        meta_station = meta_station.reset_index()
        self.longitude = meta_station["Longitude"][0]
        self.latitude = meta_station["Latitude"][0]
        self.timezone = TimezoneFinder().timezone_at(lng=self.longitude, lat=self.latitude)
        timezone_station = pytz.timezone(self.timezone)
        time_difference = timezone_station.utcoffset(datetime.now()).total_seconds() / 3600
        
        # Group data based on the timezone-adjusted hour.
        timestamp = pd.Timestamp(hour_of_the_day + time_difference, unit='h')
        self.groups = self.data.groupby(pd.Grouper(key=time_column, freq="D", origin=timestamp))
        
        self.list_of_groups = [group for _, group in self.groups if not group.empty]
        
        return self.list_of_groups

    def load_data_lmd_flatten(self, quarters_behind_of_LMD=24*4):
        """
        Optimized version of the function that flattens LMD data from the previous 24 hours into
        a single row for each day.

        :param quarters_behind_of_LMD: How many 15-minute intervals to consider from the past data.
        :return: A DataFrame with flattened LMD data, with each day represented by a single row.
        """
        # Generate column headers for flattened data.
        lmd_columns = [col for col in self.list_of_groups[0].columns if col.startswith('lmd')]
        time_keys = [timedelta for timedelta in range(-quarters_behind_of_LMD, 0)]
        flat_col_headers = [f'{col}_{t}' for col in lmd_columns for t in time_keys]
        
        # Prepare a list to hold the resulting new data rows
        new_rows = []
        
        # Loop through each day's data except the first day
        for day_num in range(1, len(self.list_of_groups)):
            previous_day_lmd_data = self.list_of_groups[day_num - 1][lmd_columns].tail(quarters_behind_of_LMD)
            flat_data_values = previous_day_lmd_data.values.flatten().tolist()
            if len(flat_data_values) != len(flat_col_headers):
                # skip
                continue  
            flat_series = pd.Series(flat_data_values, index=flat_col_headers)
            
            # Get the first row of the current day's data (without LMD)
            current_day_data_row = self.list_of_groups[day_num].drop(lmd_columns, axis=1).iloc[0]
            
            # Combine the first row of current day's data with the flattened LMD data
            combined_row = pd.concat([current_day_data_row, flat_series])
            new_rows.append(combined_row)
        
        # Combine all new rows into a single DataFrame
        self.data = pd.DataFrame(new_rows).reset_index(drop=True)
        # update the index to be the date_time column
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index=self.data["date_time"]
            self.data.index = pd.to_datetime(self.data.index)
        self.data=self.data[1:-1]
        self.data=pd.concat(self.data)
        return self.data

    
    
    def remove_night_data(self):
        count=0
        for i, data in enumerate(self.data.values):
            # Find latitude and longitude from metadata variable based on station number stored in data[15]
            api_url="https://api.sunrise-sunset.org/json?"

            # Get the date of the measurement
            current_date = data[0].date()
            previous_date=datetime(2000,1,1).date()
            # Call the API per day and not per each measurement, to save computations time
            if previous_date != current_date:
                previous_date = current_date

                params = {"lat": self.latitude, "lng": self.longitude, "date": current_date}
                # Use the API to return sunrise and sunset times based on the three parameters
                sunrise_sunset = sunrisesunset(api_url, params)

            # keep only data from sunrise til sunset, i.e. remove night data
            # use both hour and minutes
            current_time = data[0].time()
            sunrise_time = (datetime.strptime(sunrise_sunset[0],"%H:%M:%S").time())
            sunset_time = (datetime.strptime(sunrise_sunset[1],"%H:%M:%S").time())
            # Check if current time is not in between sunrise (PM) and sunset (AM) because of UTC
            if sunrise_time > current_time > sunset_time:
                # remove row i based on the Timestamp index
                # Make sure to run this only one station at a time, since there are the same
                # timestamps for each station, but located at different latitudes & longitudes
                # resulting in different sunset & sunrise
                self.data.drop(self.data.index[i-count], inplace=True)
                count = count + 1
        


    def split_data_into_train(self):
        """
        Splits the data into training and testing sets.

        :return: A tuple containing the training and testing feature sets and targets.
        """
        cols = self.data.columns.tolist()
        # drop features with date_time in the name
        cols = [col for col in cols if 'date_time' in col]
        features = self.data.drop(columns=['power']+cols)
        power = self.data['power']
        
        len_data = round(len(power) * 0.2)
        self.features_test = features[-len_data:]
        self.features = features[:-len_data]
        self.power_test = power[-len_data:]
        self.power = power[:-len_data]
        
        return self.features, self.features_test, self.power, self.power_test
def load_data_shifted(self,hoursAhead=24.0):
    data=self.data_raw.drop(columns=['nwp_pressure','lmd_pressure'])
    # The LMD named features is not known 24 hours in advance, so we shift the data 24 hours back
    for feature in data.columns:
        if feature.startswith("lmd"):
            # theres 4 samples per hour, so we shift 24*4=96 samples back
            # fx if we have lmd at 12:00 we move that measurement to the next day at 12:00
            data[feature]=data[feature].shift(hoursAhead)
    # drop the first 96 samples
    data=data.dropna()
    #print(data["lmd_totalirrad"])
    self.data=data
    return data