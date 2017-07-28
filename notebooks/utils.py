import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from haversine import haversine


cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=datetime(2015, 12, 31),
                        end=datetime(2017, 1, 1))


def distance(positions):
    lats = [(positions[0], 0.0), (positions[2], 0.0)]
    longs = [(0.0, positions[1]), (0.0, positions[3])]
    dist = (haversine(lats[0], lats[1]) + haversine(longs[0], longs[1]))/1000
    return dist

def preprocess(filepath):
    assert filepath is not None
    data = pd.read_csv(filepath, parse_dates=['pickup_datetime'])
    
    data['pickup_hr'] = data['pickup_datetime'].dt.hour
    data['pickup_min'] = data['pickup_datetime'].dt.minute
    data['pickup_sec'] = data['pickup_datetime'].dt.second
    data['pickup_day'] = data['pickup_datetime'].dt.dayofweek
    data['pickup_date'] = data['pickup_datetime'].dt.day
    data['pickup_mon'] = data['pickup_datetime'].dt.month
    
    data['quarter'] = data['pickup_datetime'].dt.quarter
    data['weekday'] = (data['pickup_day'] < 5).astype(int)
    
    data['holiday'] = (data['pickup_datetime'].dt.date.astype('datetime64').isin(holidays)).astype(int)
    
    data = data.assign(st_dist=lambda df: np.sqrt((df.pickup_longitude-df.dropoff_longitude)**2 + 
                                       (df.pickup_latitude-df.dropoff_latitude)**2))
    
    y = data[[u'pickup_latitude', u'pickup_longitude', 
          u'dropoff_latitude', u'dropoff_longitude']]
    data['h_dist'] = map(lambda pos: distance(pos), y.as_matrix())
    
    data['s_pickup_latitude'] = (data['pickup_latitude'] - data['pickup_latitude'].min())/(data['pickup_latitude'].max() - data['pickup_latitude'].min())
    data['s_pickup_longitude'] = (data['pickup_longitude'] - data['pickup_longitude'].min())/(data['pickup_longitude'].max() - data['pickup_longitude'].min())

    data['s_dropoff_latitude'] = (data['dropoff_latitude'] - data['dropoff_latitude'].min())/(data['dropoff_latitude'].max() - data['dropoff_latitude'].min())
    data['s_dropoff_longitude'] = (data['dropoff_longitude'] - data['dropoff_longitude'].min())/(data['dropoff_longitude'].max() - data['dropoff_longitude'].min())
    
    data['flag'] = (data['store_and_fwd_flag'] == 'Y').astype(int)
    
    data['month_end'] = (data['pickup_datetime'].dt.is_month_end).astype(int)
    data['month_start'] = (data['pickup_datetime'].dt.is_month_start).astype(int)
    
    return data