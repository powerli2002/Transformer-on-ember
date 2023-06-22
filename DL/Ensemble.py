
# import os
# import lief
# import numpy as np
# import ember

# from classify_DL import Classifier_DL
# import lightgbm as lgb

# total = 0
# DL_mal = 0
# DL_good = 0
# ember_mal = 0
# ember_good = 0



# def find_exe_files(folder_path):
#     exe_files = []
#     for dirpath, dirnames, filenames in os.walk(folder_path):
#         for filename in filenames:
#             if filename.endswith('.exe'):
#                 exe_files.append(os.path.join(dirpath, filename))
#     return exe_files

# save_path = "/home/lizijian/lincode/ember/DL/save_ans.txt"
# # 读取文件

# folder_path = "/home/lizijian/lincode/ember/evaluation/malware/history"
# # folder_path = "/home/lizijian/lincode/ember/evaluation/samples_goodware"
# # exe_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".exe")]

# exe_files = find_exe_files(folder_path)

# # ember初始化
# modelpath = "/home/lizijian/lincode/ember/data/ember2018/ember_model_2018.txt"
# lgbm_model = lgb.Booster(model_file=modelpath)

# # transformer初始化
# classify_DL = Classifier_DL("histogram")
# total = len(exe_files)
# with open(save_path,"w") as f:
#     for exe_file in exe_files:

        
#         # binary_path = "/home/lizijian/lincode/ember/evaluation/malware/Trojan-Downloader.Win32.Murlo.A.exe"

#         # binary_path = "./evaluation/samples_goodware/perfmon.exe"
#         # =====================深度学习模型检测==================
#         DL_ans,DL_output = classify_DL.detect_file(exe_file)

#         if(DL_ans == 1):
#             DL_mal += 1
#             f.write("transformer-检测到恶意软件！" + exe_file + str(DL_output.data) + "\n")
#             print("transformer-检测到恶意软件！" + exe_file + str(DL_output.data) + "\n")
#         elif(DL_ans == 0):
#             DL_good += 1
#             f.write("transformer-检测到良性软件！" + exe_file + str(DL_output.data) + "\n")
#             print("transformer-检测到良性软件！" + exe_file + str(DL_output.data) + "\n")

#     # ========================Ember检测=====================
#         file_data = open(exe_file, "rb").read()
#         score = ember.predict_sample(lgbm_model, file_data, 2)
#         if score  > 0.4:
#             ember_mal += 1
#             f.write("ember-检测到恶意软件！" + exe_file + "score:" + str(score) + "\n")
#             print("ember-检测到恶意软件！" + exe_file + "score:" + str(score) + "\n")
#         else:
#             ember_good += 1
#             f.write("ember-检测到良性软件！" + exe_file + "score:" + str(score) + "\n")
#             print("ember-检测到良性软件！" + exe_file + "score:" + str(score) + "\n")

# print("总共检测软件数量：" + str(total) + "\n")
# print("transformer-恶意软件数量：" + str(DL_mal) + "\n")
# # print("transformer-良性软件数量：" + str(DL_good) + "\n")
# print("transformer-检出率：" + str(DL_mal/total) + "\n")
# print("ember-恶意软件数量：" + str(ember_mal) + "\n")
# # print("ember-良性软件数量：" + str(ember_good) + "\n")
# print("ember-检出率：" + str(ember_mal/total) + "\n")

# f.close()   

import os
import lief
import numpy as np
# import ember
from utils import predict_sample
import matplotlib.pyplot as plt

from classify_DL import Classifier_DL
import lightgbm as lgb

total = 0
DL_mal = 0
DL_good = 0
ember_mal = 0
ember_good = 0

def find_exe_files(folder_path):
    exe_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.exe'):
                exe_files.append(os.path.join(dirpath, filename))
    return exe_files

save_path = "/home/lizijian/lincode/ember/DL/save_ans.txt"
# folder_path = "/home/lizijian/lincode/ember/evaluation/malware/history"
folder_path = "/home/lizijian/lincode/ember/evaluation/samples_goodware"
exe_files = find_exe_files(folder_path)

ember_scores = []
transformer_scores = []

modelpath = "/home/lizijian/lincode/ember/data/ember2018/ember_model_2018.txt"
lgbm_model = lgb.Booster(model_file=modelpath)

# classify_DL = Classifier_DL("normal")
classify_DL = Classifier_DL("histogram")
total = len(exe_files)

with open(save_path, "w") as f:
    for exe_file in exe_files:
        DL_ans, DL_output = classify_DL.detect_file(exe_file)

        if DL_ans == 1:
            DL_mal += 1
            f.write("transformer-检测到恶意软件！" + exe_file + str(DL_output.data) + "\n")
        elif DL_ans == 0:
            DL_good += 1
            f.write("transformer-检测到良性软件！" + exe_file + str(DL_output.data) + "\n")

        file_data = open(exe_file, "rb").read()
        score = predict_sample(lgbm_model, file_data, 2)
        if score > 0.5:
            ember_mal += 1
            f.write("ember-检测到恶意软件！" + exe_file + "score:" + str(score) + "\n")
        else:
            ember_good += 1
            f.write("ember-检测到良性软件！" + exe_file + "score:" + str(score) + "\n")

        ember_scores.append(score)
        transformer_scores.append(DL_output.data)

# Bar chart
labels = ["Ember", "Transformer"]
malicious_counts = [ember_mal, DL_mal]
total_counts = [total, total]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, malicious_counts, width, label='Malicious', color='red')
rects2 = ax.bar(x + width/2, total_counts, width, label='Total', color='blue')

ax.set_ylabel('Count')
ax.set_title('Detection Rates of Ember and Transformer Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig("/home/lizijian/lincode/ember/DL/detection_rates.png")  # Save the chart as an image

plt.show()

with open(save_path, "a") as f:
    f.write("\n总共检测软件数量：" + str(total) + "\n")
    f.write("transformer-恶意软件数量：" + str(DL_mal) + "\n")
    f.write("transformer-检出率：" + str(DL_mal/total) + "\n")
    f.write("ember-恶意软件数量：" + str(ember_mal) + "\n")
    f.write("ember-检出率：" + str(ember_mal/total) + "\n")

print("总共检测软件数量：" + str(total) + "\n")
print("transformer-恶意软件数量：" + str(DL_mal) + "\n")
print("transformer-检出率：" + str(DL_mal/total) + "\n")
print("ember-恶意软件数量：" + str(ember_mal) + "\n")
print("ember-检出率：" + str(ember_mal/total) + "\n")


 
        










# import os
# import lief
# import numpy as np
# import ember
# from utils import predict_sample
# import matplotlib.pyplot as plt

# from classify_DL import Classifier_DL
# import lightgbm as lgb

# total = 0
# DL_mal = 0
# DL_good = 0
# ember_mal = 0
# ember_good = 0
# ensemble_mal = 0
# ensemble_good = 0

# def find_exe_files(folder_path):
#     exe_files = []
#     for dirpath, dirnames, filenames in os.walk(folder_path):
#         for filename in filenames:
#             if filename.endswith('.exe'):
#                 exe_files.append(os.path.join(dirpath, filename))
#     return exe_files

# save_path = "/home/lizijian/lincode/ember/DL/save_ans.txt"
# # folder_path = "/home/lizijian/lincode/ember/evaluation/malware/history"
# folder_path = "/home/lizijian/lincode/ember/evaluation/samples_goodware"
# exe_files = find_exe_files(folder_path)

# ember_scores = []
# transformer_scores = []
# ensemble_scores = []

# modelpath = "/home/lizijian/lincode/ember/data/ember2018/ember_model_2018.txt"
# lgbm_model = lgb.Booster(model_file=modelpath)

# classify_DL = Classifier_DL("histogram")
# total = len(exe_files)

# with open(save_path, "w") as f:
#     for exe_file in exe_files:
#         DL_ans, DL_output = classify_DL.detect_file(exe_file)

#         if DL_ans == 1:
#             DL_mal += 1
#             f.write("transformer-检测到恶意软件！" + exe_file + str(DL_output.data) + "\n")
#         elif DL_ans == 0:
#             DL_good += 1
#             f.write("transformer-检测到良性软件！" + exe_file + str(DL_output.data) + "\n")

#         file_data = open(exe_file, "rb").read()
#         score = predict_sample(lgbm_model, file_data, 2)
#         if score > 0.5:
#             ember_mal += 1
#             f.write("ember-检测到恶意软件！" + exe_file + "score:" + str(score) + "\n")
#         else:
#             ember_good += 1
#             f.write("ember-检测到良性软件！" + exe_file + "score:" + str(score) + "\n")

#         ensemble_ans = 1 if score > 0.4 or DL_ans == 1 else 0
#         if ensemble_ans == 1:
#             ensemble_mal += 1
#             f.write("ensemble-检测到恶意软件！" + exe_file + "\n")
#         else:
#             ensemble_good += 1
#             f.write("ensemble-检测到良性软件！" + exe_file + "\n")

#         ember_scores.append(score)
#         transformer_scores.append(DL_output.data)
#         ensemble_scores.append(ensemble_ans)

# # Bar chart
# labels = ["Ember", "Transformer", "Ensemble"]
# malicious_counts = [ember_mal, DL_mal, ensemble_mal]
# total_counts = [total, total, total]

# x = np.arange(len(labels))
# width = 0.35

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, malicious_counts, width, label='Malicious', color='red')
# rects2 = ax.bar(x + width/2, total_counts, width, label='Total', color='blue')

# ax.set_ylabel('Count')
# ax.set_title('Detection Rates of Ember, Transformer, and Ensemble Models')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # Add labels on top of each bar
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# fig.tight_layout()

# plt.savefig("/home/lizijian/lincode/ember/DL/detection_rates.png",dpi = 600)  # Save the chart as an image

# plt.show()

# with open(save_path, "a") as f:
#     f.write("\n总共检测软件数量：" + str(total) + "\n")
#     f.write("transformer-恶意软件数量：" + str(DL_mal) + "\n")
#     f.write("transformer-检出率：" + str(DL_mal/total) + "\n")
#     f.write("ember-恶意软件数量：" + str(ember_mal) + "\n")
#     f.write("ember-检出率：" + str(ember_mal/total) + "\n")
#     f.write("ensemble-恶意软件数量：" + str(ensemble_mal) + "\n")
#     f.write("ensemble-检出率：" + str(ensemble_mal/total) + "\n")

# print("总共检测软件数量：" + str(total) + "\n")
# print("transformer-恶意软件数量：" + str(DL_mal) + "\n")
# print("transformer-检出率：" + str(DL_mal/total) + "\n")
# print("ember-恶意软件数量：" + str(ember_mal) + "\n")
# print("ember-检出率：" + str(ember_mal/total) + "\n")
# print("ensemble-恶意软件数量：" + str(ensemble_mal) + "\n")
# print("ensemble-检出率：" + str(ensemble_mal/total) + "\n")





