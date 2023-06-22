import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import concurrent.futures
 
class PEDataSet(Dataset):
    def __init__(self, data_path, is_train):
        if is_train:    # 判断是否是训练数据
            for i in range(len(data_path)):
                if i == 0:
                    self.x, self.y = read_json(data_path[i])
                else:
                    x, y = read_json(data_path[i])
 
                    self.x = np.vstack((self.x, x))
                    self.y = np.vstack((self.y, y))
        else:
            self.x, self.y = read_json(data_path)
 
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.len = self.x.size()[0]
 
    def __getitem__(self, index):
        return self.x[index], self.y[index]
 
    def __len__(self):
        return self.len
 
 
def read_json(json_path):
    
    df = pd.read_json(json_path, lines=True)
 
    label = df["label"]
    header = df["header"]
    general = df["general"]
    section = df["section"]
 
    x = []
    y = []
    for i in range(label.size):
        # 未标记样本,舍弃掉
        if label[i] == -1:
            continue
 
        y.append(label[i])
 
        tmp_x = []
 
        tmp_x.append(header[i]["optional"]["sizeof_code"] % 255)
        tmp_x.append(header[i]["optional"]["sizeof_headers"] % 255)
        tmp_x.append(header[i]["optional"]["sizeof_heap_commit"] % 255)
 
        tmp_x.append(general[i]["size"] % 255)
        tmp_x.append(general[i]["vsize"] % 255)
        tmp_x.append(general[i]["has_debug"] % 255)
        tmp_x.append(general[i]["has_relocations"] % 255)
        tmp_x.append(general[i]["has_resources"] % 255)
        tmp_x.append(general[i]["has_signature"] % 255)
        tmp_x.append(general[i]["has_tls"] % 255)
        tmp_x.append(general[i]["symbols"] % 255)
 
        tmp_x.append(len(section[i]["sections"]) % 255)
 
        x.append(tmp_x)
 
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
 
    return np.array(x), y




    def read_json_parallel(self, data_path, num_threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.read_json, data_path))

        x_list, y_list = zip(*results)
        x = np.vstack(x_list)
        y = np.vstack(y_list)
        return x, y