# 使用Transformer基于ember数据集的训练

This paper describes many more details about the dataset: https://arxiv.org/abs/1804.04637

## Transformer模型效果

### 第一次训练：12维输入

![1687404309141](image/README/1687404309141.png)

![1687404333498](image/README/1687404333498.png)

![1687404359602](image/README/1687404359602.png)


### 第二次训练 524维输入，加入直方图和字节熵


![1687404377319](image/README/1687404377319.png)

![1687404405166](image/README/1687404405166.png)

恶意样本

![1687404418381](image/README/1687404418381.png)



善意样本

![1687404456050](image/README/1687404456050.png)

### 第三次训练 524维特征进行归一化（存在问题）

![1687404486031](image/README/1687404486031.png)

![1687404495854](image/README/1687404495854.png)



## 融合模型

malicious if ember > 0.5 or Transformer == malware

![1687404567264](image/README/1687404567264.png)


![1687404629317](image/README/1687404629317.png)
