import zlib
import pickle
import math
import random
import numpy as np
import torch
import argparse
import os
import sys
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


def list_files(directory):
    origin_file_path = []
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.pkl'):
            origin_file_path.append(os.path.join(directory, file))

    return origin_file_path


class Argoverse_Dataset(Dataset):
    def __init__(self, args, validation=False):
        self.validation = validation
        self.ex_file_path = args.ex_file_path
        self.val_ex_file_path = args.val_ex_file_path

        # === Data Augmentations ===
        self.static_agent_drop = args.static_agent_drop
        self.scaling = args.scaling

        if validation:
            self.static_agent_drop = False
            self.scaling = False

        if validation or args.validate:
            self.ex_list = list_files(self.val_ex_file_path)
        else:
            self.ex_list = list_files(self.ex_file_path)

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        data_compress = self.ex_list[idx]
        with open(data_compress, 'rb') as f:
            data_compress = pickle.load(f)
        mapping = pickle.loads(zlib.decompress(data_compress))
        mapping = self.traditional_preprocess(mapping)
        return mapping

    def traditional_preprocess(self, mapping):
        # agent vector
        # [pre_x, pre_y, x, y, ...]
        #
        # lane vector
        # [..., y, x, pre_y, pre_x]

        matrix = mapping["matrix"]  # 所有polyline的相关数据【N * 128】
        polyline_spans = mapping["polyline_spans"] # polyline 对应的index区间， slice
        map_start_polyline_idx = mapping["map_start_polyline_idx"] # Map 和 lane 区别开的坐标
        labels = torch.from_numpy(mapping["labels"])  # 【30， 23， 2】其中这个23代表什么，map_start_polyline_index=21
        # 上述 有几个障碍物 存在历史层面没有出现， 所以讲障碍物过滤掉， 但是label不过滤 ？

        # deleted agents with 1 vector  代表23个障碍物，在t=2秒的时候，是否存在，作为一个label
        information_matrix = torch.from_numpy(mapping["information_matrix"])  # 【50 ， 23】

        # 这个地方的 含义 ？？？？？？
        vector_available = torch.where(
            (information_matrix[:20] == 1).sum(dim=0) > 1)[0]
        # 代表一个障碍物 到底存活了几zhen， 在2s内，必须要大于一帧才有意义，但是为什么要大于一帧?
        # 这个地方有点问题， 帧之前是断开的， 那维度中有时间，比例过少，应该也不好学习吧
        labels = labels[:, vector_available]  # [30, 21, 2]
        information_matrix = information_matrix[:, vector_available] # [50, 23]
        pos_matrix = torch.from_numpy(mapping["pos_matrix"])[
            :, vector_available]

        assert map_start_polyline_idx == labels.shape[1]  # 这里的校验是怎么存在的？
        assert map_start_polyline_idx == information_matrix.shape[1]

        random_scale = 1.0  # 增加系数
        if self.scaling:
            random_scale = 0.75 + random.random()*0.5

        agent_list = []
        agent_indices = []
        lane_list = []
        meta_info = []
        for j, polyline_span in enumerate(polyline_spans):
            tensor = torch.tensor(matrix[polyline_span])

            # === Augmentation ===
            drop = False
            is_agent_polyline = j < map_start_polyline_idx

            if is_agent_polyline and (len(tensor) == 1):
                # 如果障碍物 只出现一帧，那不考虑该障碍物， 不过这时候，不会出现该情况的
                continue

            if is_agent_polyline: # 计算起始点偏移，也就是通俗的一个障碍物移动了多少米
                displacement = torch.norm(tensor[-1, 2:4] - tensor[0, :2])

            # === Static Agent Drop === # 障碍物偏移不超过1m的情况，如果有辆车路边停车呢 ？？
             # 较难去 区分 静态的路边障碍物 或者  路上的真实停下的车 ？
            if (is_agent_polyline) and (j != 0) and (displacement < 1.0) and (self.static_agent_drop):
                drop = random.random() < 0.1

            if is_agent_polyline and not drop:
                tensor[:, :4] *= random_scale
                agent_list.append(tensor)
                agent_indices.append(j)

                dx, dy = tensor[-1, 2:4] - tensor[-1, :2]
                degree = math.atan2(dy, dx)
                x = tensor[-1, 2]
                y = tensor[-1, 3]
                pre_x = tensor[-1, 0]
                pre_y = tensor[-1, 1]
                info = torch.tensor(
                    [degree, x, y, pre_x, pre_y]).unsqueeze(dim=0)
                meta_info.append(info) # 当前帧的关系信息

            elif not is_agent_polyline:
                tensor[:, -4:] *= random_scale
                tensor[:, -18:-16] *= random_scale
                lane_list.append(tensor)

        assert len(agent_indices) > 0
        try:
            assert len(lane_list) > 0
        except:
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            print(f'file name: {mapping["file_name"]}')

        meta_info = torch.cat(meta_info, dim=0)
        assert len(meta_info) == len(agent_indices)

        labels *= random_scale
        labels = labels[:, torch.tensor(agent_indices, dtype=torch.long)]

        pos_matrix = pos_matrix[:, torch.tensor(
            agent_indices, dtype=torch.long)]*random_scale

        label_is_valid = information_matrix[20:, torch.tensor(
            agent_indices, dtype=torch.long)]  # 关于未来的label  [30, 21]

        information_matrix = information_matrix[:, torch.tensor(
            agent_indices, dtype=torch.long)]

        full_traj = torch.mean(information_matrix, dim=0) == 1
        # moving.shape = (#agent_num)
        moving = torch.norm(pos_matrix[19] - pos_matrix[0], dim=-1) > 6.0
        moving[0] = True

        consider = torch.where(torch.logical_and(full_traj, moving))[0]
        assert 0 in consider

        new_mapping = {"agent_data": agent_list,
                       "lane_data": lane_list,
                    #    "city_name": mapping["city_name"],
                       "file_name": mapping["file_name"],
                       "origin_labels": torch.tensor(mapping["origin_labels"],dtype=torch.float32),
                       "labels": labels,
                       "label_is_valid": label_is_valid,
                       "consider": consider,
                       "cent_x": mapping["cent_x"],
                       "cent_y": mapping["cent_y"],
                       "angle": mapping["angle"],
                       "meta_info": meta_info}

        return new_mapping


def batch_list(batch):
    return batch




