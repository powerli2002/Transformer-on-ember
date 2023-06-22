import torch
from PEModel import PEModel
from PIL import Image
from torchvision.transforms import ToTensor
from PEModel import PEModel



# 加载模型
model_path = "./model_data/cnnmodel-11-0.6621"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PEModel().to(device)
model.load_state_dict(torch.load(model_path)["model"])
model.eval()

# 加载待预测的样本数据
# 这里假设你有一个名为"sample.bin"的样本文件，你需要将其转换为模型所需的输入格式
# 例如，将样本转换为张量或使用与训练集相同的数据预处理方法
sample_path = "sample.bin"
sample_tensor = preprocess_sample(sample_path)  # 需要根据实际情况实现preprocess_sample函数

# 进行预测
with torch.no_grad():
    sample_tensor = sample_tensor.to(device)
    output = model(sample_tensor)
    _, predicted = torch.max(output.data, dim=1)

# 输出预测结果
if predicted.item() == 0:
    print("该样本被预测为恶意软件")
else:
    print("该样本被预测为正常软件")
