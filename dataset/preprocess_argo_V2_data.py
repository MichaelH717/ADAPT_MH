import math
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from enum import Enum
from pathlib import Path
import zlib
import time

import multiprocessing
from multiprocessing import Process

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file


parser = argparse.ArgumentParser("Argoverse2 Data Preprocessig")

# === Data Related Parameters ===
parser.add_argument('--data_dir', type=str, default='/data/argo2/argo2_dataset/debug')
parser.add_argument('--output_dir', type=str, default="/data/argo2/argo2_dataset/debug_output")
parser.add_argument('--core_num', type=int, default=16)
parser.add_argument('--max_dis_to_lane', type=float, default=100.0)
parser.add_argument('--max_dis_to_agent', type=float, default=100.0)
parser.add_argument('--feature_size', type=int, default=128)

parser.add_argument('--historical_step', type=int, default=20)
parser.add_argument('--future_step', type=int, default=30)

args = parser.parse_args()

TIMESTEP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
VELOCITY_X = 5
VELOCITY_Y = 6
HEADING = 7


VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

class object_type(Enum):
    VEHICLE = 1,
    PEDESTRIAN = 2,
    MOTORCYCLIST = 3,
    CYCLIST = 4,
    BUS = 5,
    UNKNOWN = 6,

class Arguments:
    def __init__(self):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.core_num = args.core_num
        self.max_dis_to_lane = args.max_dis_to_lane
        self.max_dis_to_agent = args.max_dis_to_agent
        self.feature_size = args.feature_size
        
        self.historical_step = args.historical_step
        self.future_step = args.future_step

def create_dataset(args, raw_file_name):
    if raw_file_name is None:
        return
    
    dir = os.path.join(args.data_dir, raw_file_name)
    try:
        df = pd.read_parquet(os.path.join(dir, f'scenario_{raw_file_name}.parquet'))
    except:
        print(f'an error in the corresponding file {dir}')
        return
            
    map_path = os.path.join(dir, f'log_map_archive_{raw_file_name}.json')
    map_data = read_json_file(map_path)

    centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                   for lane_segment in map_data['lane_segments'].values()}
    map_api = ArgoverseStaticMap.from_json(Path(map_path))

    mapping = dict()
    mapping['data_source'] = 'argo2'
    mapping['file_name'] = raw_file_name
    mapping['scenario_id'] = get_scenario_id(df)
    mapping['city'] = get_city(df)
    mapping['scene_step'] = [args.historical_step, args.future_step]

    get_scene_features(df, args, mapping, map_api, centerlines)
    origin_name = mapping['file_name']
    file_name = mapping['data_source'] + '_' + origin_name + '.pkl'
    data_compress = zlib.compress(pickle.dumps(mapping))

    pickle_file_name = os.path.join(args.output_dir, file_name)
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(data_compress, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def get_scenario_id(df: pd.DataFrame) -> str:
    return df['scenario_id'].values[0]

def get_city(df: pd.DataFrame) -> str:
    return df['city'].values[0]

def get_scene_features(df: pd.DataFrame, args, mapping: dict, map_api, centerlines: dict):
    new_df = df[df['timestep'] < args.historical_step]
    agent_ids = list(new_df['track_id'].unique())
    agent_df = df[df['track_id'].isin(agent_ids)]

    focal_track_id = None

    id2info = {}
    vector_num = 0
    
    # 1. get id2info
    for track_id, track_df in agent_df.groupby('track_id'):
        for _, row  in track_df.iterrows():
            if row['track_id'] == 'AV':
                focal_track_id = row['focal_track_id']
            
            # Filter out unselected future information
            if row['timestep'] >= args.future_step + args.historical_step:
                break
            
            line = [row['timestep'], row['track_id'], row['object_type'], row['position_x'], row['position_y'], \
                    row['velocity_x'], row['velocity_y'], row['heading']]      
        
            if track_id in id2info:
                id2info[track_id].append(line)
                vector_num += 1
            else:
                id2info[track_id] = [line]
    
    assert focal_track_id is not None
    
    # 2. get base_pos and base_angle
    focal_track_info = id2info[focal_track_id]
    assert len(focal_track_info) == args.historical_step + args.future_step
    cent_x = focal_track_info[args.historical_step][X]
    cent_y = focal_track_info[args.historical_step][Y]
    
    # 3. get label
    origin_labels = np.zeros([args.future_step, 2])
    for i, line in enumerate(focal_track_info[args.historical_step:]):
        origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
    mapping['origin_labels'] = origin_labels 
    
    smoothing_times = 4
    base_angle = 0.0
    for i in range(smoothing_times):
        base_angle += focal_track_info[args.historical_step - 1 - i][HEADING]   
    base_angle = base_angle / smoothing_times
    
    mapping['cent_x'] = cent_x
    mapping['cent_y'] = cent_y
    mapping['angle'] = math.radians(90) - base_angle
    mapping['focal_track_id'] = focal_track_id
    
    # 4. norm and rot
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(
                line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], mapping['angle'])
            
    pos_matrix, information_matrix, timestamps = position_matrix(id2info, args, mapping)
    mapping["pos_matrix"] = pos_matrix  # has nan
    mapping["information_matrix"] = information_matrix
    mapping["timestamps"] = timestamps
    
    # 5. get vectors,  map_start_polyline_idx, polyline_spans
    keys = get_key_list(id2info, focal_track_id)
    vectors = []
    polyline_spans = []
    his_step = args.historical_step
    for id in keys:
        info = id2info[id]
        start = len(vectors)

        for i, line in enumerate(info):
            if line[TIMESTEP] > his_step - 1:
                break
            
            x, y = line[X], line[Y]
            velocity_x, velocity_y = line[VELOCITY_X], line[VELOCITY_Y]
            if i > 0:
                vector = [line_pre[X], line_pre[Y], x, y,
                          line_pre[VELOCITY_X], line_pre[VELOCITY_Y], velocity_x, velocity_y,
                          line[TIMESTEP], 
                          line[OBJECT_TYPE] == object_type.VEHICLE,
                          line[OBJECT_TYPE] == object_type.PEDESTRIAN,
                          line[OBJECT_TYPE] == object_type.MOTORCYCLIST,
                          line[OBJECT_TYPE] == object_type.CYCLIST,
                          line[OBJECT_TYPE] == object_type.BUS,
                          len(polyline_spans), i]
                vectors.append(get_pad_vector(vector, args.feature_size))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != focal_track_id
        else:
            polyline_spans.append([start, end])
    
    mapping['map_start_polyline_idx'] = len(polyline_spans)
    
    # 6. get relative map_info
    id2laneinfo = dict()
    for lane_segment in map_api.get_scenario_lane_segments():
        lane_points_xy = centerlines[lane_segment.id].xyz[:, :2]
        
        # norm and rot
        add_lane =  True 
        for i, point in enumerate(lane_points_xy):
            point[0], point[1] = rotate(point[0] - cent_x, point[1] - cent_y, mapping['angle'])
            # long distance lane filtering
            if math.hypot(point[0], point[1]) > args.max_dis_to_lane:
                add_lane = False
                break
        
        if add_lane:
            id2laneinfo[lane_segment.id] = lane_points_xy
         
    #  lane vector
    for lane_id, polygon in id2laneinfo.items():
        start = len(vectors)
        for lane_segment in map_api.get_scenario_lane_segments():
            if lane_segment.id == lane_id:
                break
            
        assert len(polygon) >= 2
        for i, point in enumerate(polygon):
            if i > 0:
                vector = [0] * args.feature_size
                vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                vector[-5] = 1  # lane center type
                vector[-6] = i
                vector[-7] = len(polyline_spans)
                vector[-8] = 0  # unused flag
                vector[-9] = 0  # unused flag
                vector[-10] = 1 if lane_segment.is_intersection else -1
                point_pre_pre = (
                    2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                if i >= 2:
                    point_pre_pre = polygon[i - 2]
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]

                vectors.append(vector)
            point_pre = point

        end = len(vectors)
        if start < end:
            polyline_spans.append([start, end])
            
    matrix = np.array(vectors)
    labels, label_is_valid = get_labels(args, id2info, mapping)
    mapping.update(dict(
        matrix=matrix,
        labels=labels,
        label_is_valid=label_is_valid,
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
    ))
    

def get_labels(args, id2info, mapping):
    labels = []
    pred_index = args.historical_step

    pos_matrix = mapping["pos_matrix"]
    information_matrix = mapping["information_matrix"]

    labels = np.zeros((args.future_step, pos_matrix.shape[1], 2)) # [future_step, agent_size, xy]
    label_is_valid = np.zeros((args.future_step, pos_matrix.shape[1]))

    for agent_id in range(pos_matrix.shape[1]):
        labels[:, agent_id] = pos_matrix[pred_index:, agent_id]
        label_is_valid[:, agent_id] = information_matrix[pred_index:, agent_id]

    labels[np.isnan(labels)] = -666
    return labels, label_is_valid

def get_pad_vector(li, feature_size):
    assert len(li) <= feature_size
    li.extend([0] * (feature_size - len(li)))
    return li

def get_key_list(id2info, focal_track_id):
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert focal_track_id in keys
    keys.remove('AV')
    keys.remove(focal_track_id)
    keys = [focal_track_id, 'AV'] + keys
    return keys

def position_matrix(id2info, args, mapping):
    global TIMESTEP, X, Y
    # position matrix shows the time vs location of agents in the scene
    # rows are agents, columns are time steps
    # M_ij corresponds to location of agent j in time step i
    
    agent_size = len(id2info)
    scene_step = args.historical_step + args.future_step
    focal_track_id = mapping['focal_track_id']
    timestamps = np.zeros((scene_step,))
    pos_matrix = np.ones((scene_step, agent_size, 2)) * np.nan
    information_matrix = np.zeros((scene_step, agent_size))

    keys = get_key_list(id2info, focal_track_id)

    for i, line_info in enumerate(id2info[focal_track_id]):
        ts = line_info[TIMESTEP]
        timestamps[i] = ts

    for agent_id, key in enumerate(keys):
        agent = id2info[key]
        for line_info in agent:
            for i, ts in enumerate(timestamps):
                if line_info[TIMESTEP] == ts:
                    pos_matrix[i, agent_id, 0] = line_info[X]
                    pos_matrix[i, agent_id, 1] = line_info[Y]
                    information_matrix[i, agent_id] = 1
                    break

    assert ((np.isnan(pos_matrix)).sum() == (information_matrix == 0).sum() * 2)
    return pos_matrix, information_matrix, timestamps       

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y       

def single_data_processing(queue, args):
    while True:
        raw_file = queue.get()
        if raw_file is None:
            break
        
        create_dataset(args, raw_file)
        
        
def multi_data_processing(args):
    pbar = tqdm(total=len(os.listdir(args.data_dir)))
    queue = multiprocessing.Queue(args.core_num)
    processes = [Process(target=single_data_processing, args=(
        queue, args,)) for _ in range(args.core_num)]

    for each in processes:
        each.start()

    for raw_file in os.listdir(args.data_dir):
        assert raw_file is not None
        queue.put(raw_file)
        pbar.update(1)

    while not queue.empty():
        pass

    pbar.close()

    for _ in range(args.core_num):
        queue.put(None)
    for each in processes:
        each.join()

if __name__ == '__main__':
    args = Arguments()
    start = time.time()
    multi_data_processing(args)
    end = time.time()
    print(f'data_processing need time: {end - start}')