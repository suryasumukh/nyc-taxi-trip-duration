import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock


def preprocess(filepath):
    assert filepath is not None
    data = pd.read_csv(filepath, parse_dates=['pickup_datetime'])
    
    data['pickup_hr'] = map(lambda dt: dt.hour, data['pickup_datetime'])
    data['pickup_min'] = map(lambda dt: dt.minute, data['pickup_datetime'])
    data['pickup_sec'] = map(lambda dt: dt.second, data['pickup_datetime'])
    data['pickup_day'] = map(lambda dt: dt.dayofweek, data['pickup_datetime'])
    data['pickup_date'] = map(lambda dt: dt.day, data['pickup_datetime'])
    data['pickup_mon'] = map(lambda dt: dt.month, data['pickup_datetime'])
    data['pickup_yr'] = map(lambda dt: dt.year, data['pickup_datetime'])
    
    data = data.assign(st_dist=lambda df: np.sqrt((df.pickup_longitude-df.dropoff_longitude)**2 + 
                                       (df.pickup_latitude-df.dropoff_latitude)**2))
    
    y = data[[u'pickup_longitude', u'pickup_latitude', u'dropoff_longitude', u'dropoff_latitude']]
    data['cityblock'] = map(lambda x: cityblock(x[0:2], x[2:]), y.as_matrix())
    data['s_pickup_latitude'] = (data['pickup_latitude'] - data['pickup_latitude'].min())/(data['pickup_latitude'].max() - data['pickup_latitude'].min())
    data['s_pickup_longitude'] = (data['pickup_longitude'] - data['pickup_longitude'].min())/(data['pickup_longitude'].max() - data['pickup_longitude'].min())

    data['s_dropoff_latitude'] = (data['dropoff_latitude'] - data['dropoff_latitude'].min())/(data['dropoff_latitude'].max() - data['dropoff_latitude'].min())
    data['s_dropoff_longitude'] = (data['dropoff_longitude'] - data['dropoff_longitude'].min())/(data['dropoff_longitude'].max() - data['dropoff_longitude'].min())
    
    data['flag'] = (data['store_and_fwd_flag'] == 'Y').astype(int)
    
    return data