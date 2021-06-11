# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import Counter


def epoch2human(epoch):
    return datetime.fromtimestamp(int(epoch)).strftime('%Y-%m-%d')


'''
taxi_zone_geodata = pd.read_csv('data/taxi_dataset/taxi_geodata.csv')
manhattan = taxi_zone_geodata.loc[taxi_zone_geodata['borough'] == "Manhattan"]
manhattan = manhattan.loc[manhattan['zone'] != "Governor's Island/Ellis Island/Liberty Island"]
manhattan_ids = list(manhattan.locationid)

ent2id = dict()
for ent in manhattan_ids:
    ent2id[ent] = len(ent2id)

taxi_info = pd.read_csv('data/taxi_dataset/green_tripdata_2019-01.csv')
taxi_info = taxi_info.loc[taxi_info['VendorID'] == 1]
taxi_info = taxi_info.loc[taxi_info['PULocationID'].isin(manhattan_ids)]
taxi_info = taxi_info.loc[taxi_info['DOLocationID'].isin(manhattan_ids)]
taxi_info = taxi_info.loc[taxi_info['passenger_count'] > 0]
#taxi_info = taxi_info.loc[taxi_info['trip_distance'] >= 0.1]
#taxi_info = taxi_info.loc[taxi_info['RatecodeID'] == 1]
#taxi_info = taxi_info.loc[taxi_info['payment_type'] == 2]

PUtime, DOtime, PULocationID, DOLocationID, PUcoordinate, DOcoordinate = [], [], [], [], [], []

PUtime = [datetime.strptime(i, '%m/%d/%Y %H:%M').timestamp() for i in list(taxi_info.lpep_pickup_datetime)]
DOtime = [datetime.strptime(i, '%m/%d/%Y %H:%M').timestamp() for i in list(taxi_info.lpep_dropoff_datetime)]
PULocationID = list(taxi_info.PULocationID)
DOLocationID = list(taxi_info.DOLocationID)

PUcoordinate = [(taxi_zone_geodata.loc[taxi_zone_geodata['locationid'] == i]['X'].tolist()[0], 
                 taxi_zone_geodata.loc[taxi_zone_geodata['locationid'] == i]['Y'].tolist()[0]) for i in PULocationID]
DOcoordinate = [(taxi_zone_geodata.loc[taxi_zone_geodata['locationid'] == i]['X'].tolist()[0], 
                 taxi_zone_geodata.loc[taxi_zone_geodata['locationid'] == i]['Y'].tolist()[0]) for i in DOLocationID]      

manhattan_taxi_info = pd.DataFrame(
    {'PUtime': PUtime,
     'DOtime': DOtime,
     'PULocationID': [ent2id[i] for i in PULocationID],
     'DOLocationID': [ent2id[i] for i in DOLocationID],
     'PUcoordinate': PUcoordinate,
     'DOcoordinate': DOcoordinate
    })
manhattan_taxi_info = manhattan_taxi_info.sort_values('DOtime')


#manhattan_taxi_info.to_json('data/taxi_dataset/taxi_info.json', orient='records', lines=True)

taxi_info = pd.read_json("data/taxi_dataset/taxi_info.json", lines=True)
dropof_time = [int(epoch2human(i).split('-')[-1]) for i in list(taxi_info.DOtime)]
timestamp = list(taxi_info.DOtime)

standard_time = []
for i in range(1,32):
    if i < 10:
        standard_time.append(datetime.fromisoformat("2019-01-0{}".format(i)).timestamp())
    else:
        standard_time.append(datetime.fromisoformat("2019-01-{}".format(i)).timestamp())
        
timestamps = []
for i, x in enumerate(timestamp):
    timestamps.append((x-standard_time[dropof_time[i]-1])/86400.0)

dropof_time = [i-1 for i in dropof_time]
PUlocation = list(taxi_info.PULocationID)
DOlocation = list(taxi_info.DOLocationID)
PUcoord = [tuple(i[::-1]) for i in list(taxi_info.PUcoordinate)]
DOcoord = [tuple(i[::-1]) for i in list(taxi_info.DOcoordinate)]


with open("data/taxi_dataset/taxi_data.txt", "w") as text_file:
    for i, x in enumerate(dropof_time):
        text_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x, PUlocation[i], DOlocation[i], timestamps[i], PUcoord[i][0], PUcoord[i][1], DOcoord[i][0], DOcoord[i][1]))
'''

'''
user_checkin = pd.read_csv('data/checkin_dataset/dataset_TSMC2014_TKY.csv')

standard_time = []
for i in range(31):
    if i+1 < 10:
        standard_time.append(datetime.fromisoformat("2012-05-0{}".format(i+1)).timestamp())
    else:
        standard_time.append(datetime.fromisoformat("2012-05-{}".format(i+1)).timestamp())

user_checkin = user_checkin.loc[80592:174599]

locationID = list(user_checkin.venueId)
counter = Counter(locationID)
locationID = [i[0] for i in counter.most_common(70)]

user_checkin = user_checkin.loc[user_checkin['venueId'].isin(locationID)].reset_index(drop=True)
ent2id = dict()
for ent in locationID:
    ent2id[ent] = len(ent2id)

coords = list(zip(list(user_checkin.latitude), list(user_checkin.longitude)))

userid = list(user_checkin.userId)
user_dataframe = []
for i in userid:
    user_dataframe.append(user_checkin.loc[user_checkin['userId'] == i])

tokyo_trip_data = []
for j, x in enumerate(user_dataframe):
    venueID = [ent2id[i] for i in list(x.venueId)]
    coordinate = list(zip(list(x.latitude), list(x.longitude)))
    time = [i.replace("+0000 ", '') for i in list(x['utcTimestamp'])]
    time = [datetime.strptime(i, '%a %b %d %H:%M:%S %Y').timestamp() for i in time]
    
    for k in range(len(venueID)-1):
        tokyo_trip_data.append([venueID[k], venueID[k+1], time[k+1], coordinate[k], coordinate[k+1]])
    
    if j % 200 == 0:
        print("Done, {} of {}".format(j, len(user_dataframe)))

tokyo_trip_set = set(map(tuple,tokyo_trip_data))  #need to convert the inner lists to tuples so they are hashable
tokyo_trip_data = list(map(list,tokyo_trip_set))
tokyo_trip_data.sort(key=lambda x: int(x[2]))

new_time = [i[2] for i in tokyo_trip_data]
new_time_slot = [int(epoch2human(j).split('-')[-1]) for j in new_time]

timestamps = []
for i, x in enumerate(new_time):
    timestamps.append((x-standard_time[new_time_slot[i]-1])/86400.0)


with open("data/checkin_dataset/output.txt", "w") as text_file:
    for i, x in enumerate(new_time_slot):
        text_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x, tokyo_trip_data[i][0], tokyo_trip_data[i][1], timestamps[i], tokyo_trip_data[i][3][0], tokyo_trip_data[i][3][1], tokyo_trip_data[i][4][0], tokyo_trip_data[i][4][1]))
'''
time_budget = 1-1e-7

citation_net = pd.read_json("data/authorship_dataset/data/graph_property.json")
years = list(citation_net[0])
temporal_edges = []
for i in range(len(citation_net)):
    temp_net = citation_net.iloc[i][1]
    #nodes.extend(temp_net['node'])
    temp_node = dict(temp_net['node'])
    temp_edge = temp_net['edge']
    
    for j, x in enumerate(temp_edge):
        temp_time = (j+1)*(time_budget/len(temp_edge))
        for k in x[1]:
            temporal_edges.append([i+1, x[0], k, temp_time, temp_node[x[0]], temp_node[k]])

nodes = []
for i in temporal_edges:
    nodes.append(i[1:3])

unique_nodes = np.unique(nodes)
ent2id = dict()
for ent in unique_nodes:
    ent2id[ent] = len(ent2id)
'''
with open("data/authorship_dataset/authorship_data_1.txt", "w") as text_file:
    for i, x in enumerate(temporal_edges):
        text_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x[0], ent2id[x[1]], ent2id[x[2]], x[3], x[4][0], x[4][1], x[5][0], x[5][1]))
#nodes = sorted(nodes, key=lambda x: x[0])
'''