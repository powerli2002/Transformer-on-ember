import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import concurrent.futures
from Configure import Configure
import utils
from features import PEFeatureExtractor 
# from features import process_raw_features


class PEDataSet(Dataset):

    
    
    def __init__(self, data_path, is_train):
        self.conf = Configure()
        if is_train:    # 判断是否是训练数据
            for i in range(len(data_path)):
                print("运行到第" + str(i) + "个jsonl文件")
                if i == 0:
                    self.x, self.y = self.read_json(data_path[i])
                else:
                    x, y = self.read_json(data_path[i])
 
                    self.x = np.vstack((self.x, x))
                    self.y = np.vstack((self.y, y))
        else:
            self.x, self.y = self.read_json(data_path)
 
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.len = self.x.size()[0]
 
    def __getitem__(self, index):
        return self.x[index], self.y[index]
 
    def __len__(self):
        return self.len

    # def process_raw_features(self, raw_obj):
    #     feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
    #     return np.hstack(feature_vectors).astype(np.float32)    
    
    
    # def process_raw_features(self, raw_obj):
    #     # 实例化特征提取器
    #     fe = FeatureExtractor()

    #     # 创建特征列表
    #     feature_list = []

    #     # 处理 strings 特征
    #     strings_feature = fe.process_strings_feature(raw_obj["strings"])
    #     feature_list.append(strings_feature)

    #     # 处理 general 特征
    #     general_feature = fe.process_general_feature(raw_obj["general"])
    #     feature_list.append(general_feature)

    #     # 处理 header 特征
    #     header_feature = fe.process_header_feature(raw_obj["header"])
    #     feature_list.append(header_feature)

    #     # 处理 section 特征
    #     section_feature = fe.process_section_feature(raw_obj["section"])
    #     feature_list.append(section_feature)

    #     # 处理 imports 特征
    #     imports_feature = fe.process_imports_feature(raw_obj["imports"])
    #     feature_list.append(imports_feature)

    #     # 处理 exports 特征
    #     exports_feature = fe.process_exports_feature(raw_obj["exports"])
    #     feature_list.append(exports_feature)

    #     # 处理 datadirectories 特征
    #     datadirectories_feature = fe.process_datadirectories_feature(raw_obj["datadirectories"])
    #     feature_list.append(datadirectories_feature)

    #     # 将特征向量水平连接
    #     feature_vector = np.hstack(feature_list).astype(np.float32)

    #     return feature_vector

    def process_raw_features(self, raw_obj):
        fe  = PEFeatureExtractor()
        # 使用 features.py 中的函数来处理原始特征数据
        feature_vector = fe.process_raw_features(raw_obj)
        return feature_vector
 
    def read_json(self,json_path):


        if (self.conf.model_type == "trans4"):

            df = pd.read_json(json_path, lines=True)
        
            label = df["label"]
            histogram = df["histogram"]
            byteentropy = df["byteentropy"]

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


                # a1 = max(histogram[i])
                # a2 = min(histogram[i])
                # b1 = max(byteentropy[i])
                # b2 = min(byteentropy[i])
                # a = a1-a2
                # b = b1-b2
                # for j in range (len(histogram[i])):
                #     x_normalized1 = (histogram[i][j] - a2) * (255 / a)
                #     tmp_x.append(x_normalized1)
                # for k in range(len(byteentropy[i])):
                #     x_normalized2 = (byteentropy[i][k] -b2) * (255 / b)
                #     tmp_x.append(x_normalized2)
                
                for j in range(len(histogram[i])):
                    tmp_x.append(histogram[i][j])

                for k in range(len(byteentropy[i])):
                    tmp_x.append(byteentropy[i][k])

        
                x.append(tmp_x)
            
        elif (self.conf.model_type == "trans5"):
            df = pd.read_json(json_path, lines=True)

            # 创建特征列表和标签列表
            x = []
            y = []

        # 处理每个样本
            for _, row in df.iterrows():
                if row["label"] == -1:
                    continue
                feature_vector = self.process_raw_features(row)
                x.append(feature_vector)


        
                # y.append(label[i])
                y.append(row["label"])

   
            # df = pd.read_json(json_path, lines=True)
            # self.process_raw_features(df)
            # label = df["label"]
            # histogram = df["histogram"]
            # byteentropy = df["byteentropy"]

            # header = df["header"]
            # general = df["general"]
            # section = df["section"]
            # x = []
            # y = []
            
            # for i in range(label.size):
            #     # 未标记样本,舍弃掉
            #     if label[i] == -1:
            #         continue
        
            #     y.append(label[i])
        
            #     tmp_x = []
        
            #     # tmp_x.append(header[i]["optional"]["sizeof_code"])
            #     # tmp_x.append(header[i]["optional"]["sizeof_headers"])
            #     # tmp_x.append(header[i]["optional"]["sizeof_heap_commit"])
        
            #     # tmp_x.append(general[i]["size"])
            #     # tmp_x.append(general[i]["vsize"] )
            #     # tmp_x.append(general[i]["has_debug"])
            #     # tmp_x.append(general[i]["has_relocations"])
            #     # tmp_x.append(general[i]["has_resources"])
            #     # tmp_x.append(general[i]["has_signature"])
            #     # tmp_x.append(general[i]["has_tls"])
            #     # tmp_x.append(general[i]["symbols"] )
        
            #     # tmp_x.append(len(section[i]["sections"]))
                



            #     # 字节直方图和字节熵
            #     sum1 =  sum(histogram[i])
            #     sum2 =  sum(byteentropy[i])
            #     for j in range(len(histogram[i])):
            #         tmp_x.append(histogram[i][j] / sum1)

            #     for k in range(len(byteentropy[i])):
            #         tmp_x.append(byteentropy[i][k] /  sum2)


            #     # string
            #     feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in df]



            #     x.append(tmp_x)


 

            # df = pd.read_json(json_path, lines=True)
        
            # label = df["label"]
            # histogram = df["histogram"]
            # byteentropy = df["byteentropy"]
            # strings = df["strings"]
            # general = df["general"]
            # header = df["header"]
            # section = df["section"]
            # imports = df["imports"]
            # exports = df["exports"]
            # datadirectories = df["datadirectories"]
            

            # header = df["header"]
            # general = df["general"]
            # section = df["section"]
            # x = []
            # y = []
            
            # for i in range(label.size):
            #     # 未标记样本,舍弃掉
            #     if label[i] == -1:
            #         continue
        
            #     y.append(label[i])
        
            #     tmp_x = []
        
            #     tmp_x.append(header[i]["optional"]["sizeof_code"] % 255)
            #     tmp_x.append(header[i]["optional"]["sizeof_headers"] % 255)
            #     tmp_x.append(header[i]["optional"]["sizeof_heap_commit"] % 255)
        
            #     tmp_x.append(general[i]["size"] % 255)
            #     tmp_x.append(general[i]["vsize"] % 255)
            #     tmp_x.append(general[i]["has_debug"] % 255)
            #     tmp_x.append(general[i]["has_relocations"] % 255)
            #     tmp_x.append(general[i]["has_resources"] % 255)
            #     tmp_x.append(general[i]["has_signature"] % 255)
            #     tmp_x.append(general[i]["has_tls"] % 255)
            #     tmp_x.append(general[i]["symbols"] % 255)
        
            #     tmp_x.append(len(section[i]["sections"]) % 255)





    
 


        elif(self.conf.model_type == "trans6"):


            df = pd.read_json(json_path, lines=True)
            # self.process_raw_features(df)
            label = df["label"]
            histogram = df["histogram"]
            byteentropy = df["byteentropy"]

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
        
                tmp_x.append(header[i]["optional"]["sizeof_code"])
                tmp_x.append(header[i]["optional"]["sizeof_headers"])
                tmp_x.append(header[i]["optional"]["sizeof_heap_commit"])
        
                tmp_x.append(general[i]["size"])
                tmp_x.append(general[i]["vsize"] )
                tmp_x.append(general[i]["has_debug"])
                tmp_x.append(general[i]["has_relocations"])
                tmp_x.append(general[i]["has_resources"])
                tmp_x.append(general[i]["has_signature"])
                tmp_x.append(general[i]["has_tls"])
                tmp_x.append(general[i]["symbols"] )
        
                tmp_x.append(len(section[i]["sections"]))
                



                # 字节直方图和字节熵
                sum1 =  sum(histogram[i])
                sum2 =  sum(byteentropy[i])
                for j in range(len(histogram[i])):
                    tmp_x.append(histogram[i][j] / sum1)

                for k in range(len(byteentropy[i])):
                    tmp_x.append(byteentropy[i][k] /  sum2)


                # string
                # feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in df]



                x.append(tmp_x)


        y = np.array(y)
        y = y.reshape(y.shape[0], 1)
    
        return np.array(x), y        

    # def read_json_parallel(self, data_path, num_threads):
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #         results = list(executor.map(self.read_json, data_path))

    #     x_list, y_list = zip(*results)
    #     x = np.vstack(x_list)
    #     y = np.vstack(y_list)
    #     return x, y