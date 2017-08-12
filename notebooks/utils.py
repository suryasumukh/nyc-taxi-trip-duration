import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from haversine import haversine
import pickle


cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=datetime(2015, 12, 31),
                        end=datetime(2017, 1, 1))


def haversine_dist(pos):
    p1 = (pos[0], pos[1])
    p2 = (pos[2], pos[1])
    p3 = (pos[2], pos[3])
    dist = haversine(p1, p2) + haversine(p2, p3)
    return dist

def bearing(pos):
    lat1 = np.radians(pos[0])
    lat2 = np.radians(pos[2])
    diffLong = np.radians(pos[1] - pos[3])
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
            * np.cos(lat2) * np.cos(diffLong))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

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
    
    data['holiday'] = (data['pickup_datetime'].dt.date.astype('datetime64[ns]').isin(holidays)).astype(int)
    
    data = data.assign(st_dist=lambda df: np.sqrt((df.pickup_longitude-df.dropoff_longitude)**2 + 
                                       (df.pickup_latitude-df.dropoff_latitude)**2))
    
    y = data[[u'pickup_latitude', u'pickup_longitude', 
          u'dropoff_latitude', u'dropoff_longitude']]
    data['h_dist'] = map(lambda pos: haversine_dist(pos), y.as_matrix())
    data['bearing'] = map(lambda pos: bearing(pos), y.as_matrix())
    
    data['s_pickup_latitude'] = (data['pickup_latitude'] - data['pickup_latitude'].min())/(data['pickup_latitude'].max() - data['pickup_latitude'].min())
    data['s_pickup_longitude'] = (data['pickup_longitude'] - data['pickup_longitude'].min())/(data['pickup_longitude'].max() - data['pickup_longitude'].min())

    data['s_dropoff_latitude'] = (data['dropoff_latitude'] - data['dropoff_latitude'].min())/(data['dropoff_latitude'].max() - data['dropoff_latitude'].min())
    data['s_dropoff_longitude'] = (data['dropoff_longitude'] - data['dropoff_longitude'].min())/(data['dropoff_longitude'].max() - data['dropoff_longitude'].min())
    
    data['flag'] = (data['store_and_fwd_flag'] == 'Y').astype(int)
    
    data['month_end'] = (data['pickup_datetime'].dt.is_month_end).astype(int)
    data['month_start'] = (data['pickup_datetime'].dt.is_month_start).astype(int)
    
    cluster = pickle.load(open('cluster.p', 'rb'))
    data['pickup_cluster_label'] = cluster.predict(data[['pickup_longitude', 'pickup_latitude']])
    data['dropoff_cluster_label'] = cluster.predict(data[['dropoff_longitude', 'dropoff_latitude']])
    
    idx = data['pickup_cluster_label'].as_matrix()
    data['pickup_cluster_longitude'] = cluster.cluster_centers_[idx][:, 0]
    data['pickup_cluster_latitude'] = cluster.cluster_centers_[idx][:, 1]

    idx = data['dropoff_cluster_label'].as_matrix()
    data['dropoff_cluster_longitude'] = cluster.cluster_centers_[idx][:, 0]
    data['dropoff_cluster_latitude'] = cluster.cluster_centers_[idx][:, 1]
    
    y = data[['pickup_cluster_latitude', 'pickup_cluster_longitude', 
                  'dropoff_cluster_latitude', 'dropoff_cluster_longitude']]
    data['cluster_dist'] = map(lambda pos: haversine_dist(pos), y.as_matrix())
    
    data['s_pickup_cluster_latitude'] = (data['pickup_cluster_latitude'] - data['pickup_cluster_latitude'].min())/(data['pickup_cluster_latitude'].max() - data['pickup_cluster_latitude'].min())
    data['s_pickup_cluster_longitude'] = (data['pickup_cluster_longitude'] - data['pickup_cluster_longitude'].min())/(data['pickup_cluster_longitude'].max() - data['pickup_cluster_longitude'].min())

    data['s_dropoff_cluster_latitude'] = (data['dropoff_cluster_latitude'] - data['dropoff_cluster_latitude'].min())/(data['dropoff_cluster_latitude'].max() - data['dropoff_cluster_latitude'].min())
    data['s_dropoff_cluster_longitude'] = (data['dropoff_cluster_latitude'] - data['dropoff_cluster_latitude'].min())/(data['dropoff_cluster_latitude'].max() - data['dropoff_cluster_latitude'].min())
    return data