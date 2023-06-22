
import os
import ember
import argparse
import numpy as np
from features import PEFeatureExtractor
from Transformermodel import TransformerModel
from Transformermodel_histogram import TransformerModel as TransformerModel_histogram 
import torch
from Configure import Configure

class Classifier_DL:

    def __init__(self , model_type="normal"):

        conf = Configure()

        if(model_type == "normal"):
            self.type = "normal"
            self.path_to_model = conf.model_normal
            self.modeler = TransformerModel()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(torch.cuda.get_device_name(self.device))
            self.modeler.to(self.device)
            

        #    optimizer = torch.optim.Adam(modeler.parameters(), lr=conf.lr,weight_decay=conf.decay)

            self.modeler = self.load_model(self.modeler, self.path_to_model)
            self.modeler.eval()

        elif(model_type == "histogram"):
            self.type = "histogram"
            self.path_to_model = conf.test_load_model_path_histogram
            # self.path_to_model = "/home/lizijian/lincode/ember/model_data/transformer3-model-11-0-0.3634"
            self.modeler = TransformerModel_histogram()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(torch.cuda.get_device_name(self.device))
            

            self.modeler = self.load_model(self.modeler, self.path_to_model)
            self.modeler.eval()
            self.modeler.to(self.device)


    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])


        return model


    def transfer(self,binary_path):
        feature_version = 2
        file_data = open(binary_path, "rb").read()
        extractor = PEFeatureExtractor(feature_version)
        features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
        return features
        # 这里返回的feature是2381个在0-1之间的数值。


    def handle_features(self,features):

        if self.type == "normal":
            # 在这里将features 归一化到0-255输入模型。
        # features: [histogram(256), byteentropy(256), strings(104), general(10), header(62), section(255), imports(1280), exports(128), datadirectories(30)]
            #            0         256              512             616          626         688         943          2223             2351               2381


            tmp_x = []
            # tmp_x.append(features[4] % 255)
            # 打印feature[0]的长度 
            # header
            
            
            tmp_x.append(features[685] % 255)  # sizeof_code
            tmp_x.append(features[686] % 255)  # sizeof_headers
            tmp_x.append(features[687] % 255)  # sizeof_heap_commit

            tmp_x.append(features[616] % 255) # size
            tmp_x.append(features[617] % 255) # vsize
            tmp_x.append(features[618] % 255) # has_debug
            tmp_x.append(features[619] % 255) # has_relocations
            tmp_x.append(features[620] % 255) # has_resources
            tmp_x.append(features[621] % 255) # has_signature
            tmp_x.append(features[622] % 255) # has_tls
            tmp_x.append(features[623] % 255) # symbols




            # section
            # print(features[688])
            tmp_x.append(features[688] % 255) # sections




        elif self.type == "histogram":
            tmp_x = []
            tmp_x.append(features[685] % 255)  # sizeof_code
            tmp_x.append(features[686] % 255)  # sizeof_headers
            tmp_x.append(features[687] % 255)  # sizeof_heap_commit

            tmp_x.append(features[616] % 255) # size
            tmp_x.append(features[617] % 255) # vsize
            tmp_x.append(features[618] % 255) # has_debug
            tmp_x.append(features[619] % 255) # has_relocations
            tmp_x.append(features[620] % 255) # has_resources
            tmp_x.append(features[621] % 255) # has_signature
            tmp_x.append(features[622] % 255) # has_tls
            tmp_x.append(features[623] % 255) # symbols

            tmp_x.append(features[688] % 255) # sections

            
            # a1 = max(features[0:256])
            # a2 = min(features[0:256])
            # a = a1 - a2
            # b1 = max(features[256:512])
            # b2 = min(features[256:512])
            # b = b1 - b2
            # for i in range(256):

            #     # x_normalized1 = (features[i] - a2) * (255 / a)
            #     tmp_x.append(features[i]*10000 % 255)
            #     # tmp_x.append(x_normalized1)

            # for i in range(256,512):
            #     # x_normalized2 = (features[i] - b2) * (255 / b)
            #     # tmp_x.append(x_normalized2)
            #     tmp_x.append(features[i]*100000 % 255)    
# ==============================================================
            # for i in range(256):
            #     tmp_x.append(features[i]*1000 % 255)

            # for i in range(256,512):
            #     tmp_x.append(features[i]*10000 % 255)        

            # for i in range(256):
            #     tmp_x.append(features[i]*255)

            # for i in range(256,512):
            #     tmp_x.append(features[i]*255)           


        # a =  (max(histogram[i]) - min(histogram[i]))
        # b =  (max(byteentropy[i]) - min(byteentropy[i]))
        # for j in range (len(histogram[i])):
        #     x_normalized1 = (histogram[i][j] - min(histogram[i])) * (255 / a)
        #     tmp_x.append(x_normalized1)
        # for k in range(len(byteentropy[i])):
        #     x_normalized2 = (byteentropy[i][k] - min(byteentropy[i])) * (255 / b)
        #     tmp_x.append(x_normalized2)


        # for i in range(len(tmp_x)):
        #     print(tmp_x[i])


        return tmp_x





    def classify_transformer(self,inputs):

        
        inputs = torch.tensor(inputs)
        
        inputs = inputs.to(self.device)
        inputs = inputs.unsqueeze(dim=0)
        inputs = inputs.unsqueeze(dim=1)
        outputs = self.modeler(inputs)
        _, predicted = torch.max(outputs.data, 1)

        return predicted.item(),outputs



    def detect_file(self,binary_path):
        feature = self.transfer(binary_path)
        handled_features = self.handle_features(feature)
        ans,output = self.classify_transformer(handled_features)

        return ans,output
        
        




if __name__ == "__main__":

    classifier_DL = Classifier_DL()

    # binary_path = "/home/lizijian/lincode/ember/evaluation/malware/Trojan-Downloader.Win32.Murlo.A.exe"

    binary_path = "./evaluation/samples_goodware/perfmon.exe"
    feature = classifier_DL.transfer(binary_path)
    handled_features = classifier_DL.handle_features(feature)
    ans,output =classifier_DL.classify_transformer(handled_features)
    if(ans == 1):
        print("检测到恶意软件！")
    elif(ans == 0):
        print("检测到正常软件！")
    else:
        print("出现未知错误！")

    print(ans)
    print(output)

    # print(feature)
