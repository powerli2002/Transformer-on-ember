from Configure import Configure
from PEModel import PEModel
from PEDataSet import PEDataSet
import os
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from datetime import datetime
from Transformermodel import TransformerModel

is_transformer = 1



def train(epoch,start_batch):
    total_loss = 0
    iter = 0
    total = 0
    correct = 0



    # for batch_idx, data in enumerate(train_loader, 0):
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{conf.epochs}', leave=True,initial=start_batch)
    # print(list(progress_bar))
    for batch_idx, data in enumerate(progress_bar,start=start_batch+1):
        if batch_idx  <= len(progress_bar):
            optimizer.zero_grad()  # 梯度清0

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if is_transformer:
                inputs = inputs.unsqueeze(dim=1)  # 将输入张量的维度从 (batch_size, input_dim) 转换为 (batch_size, seq_len=1, input_dim)

    
            y_pred = modeler(inputs)                                    # 前向传播
            labels = labels.squeeze(dim=1)
            loss = torch.nn.functional.cross_entropy(y_pred, labels)    # 计算损失 
            loss.backward()  # 反向传播
            
            optimizer.step()  # 梯度更新

            _, predicted = torch.max(y_pred.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            iter = iter+1
            total_loss = total_loss + loss

            
            accuracy = correct / total    
            if batch_idx % 1000 == 1:
                with open('training_log.txt', 'a') as file:
                    # Write the epoch number and loss to the file
                    if is_transformer:
                        file.write("transformer2-epoch=%d,batch_idx=%d, loss=%f,acc=%s\n" % (epoch,batch_idx, loss.item(),str(round(accuracy,4))))
                    else:
                        file.write("epoch=%d, loss=%f,acc=%s\n" % (epoch, loss.item(),str(round(accuracy,4))))
            # if batch_idx % 2 == 1:
            #     save_model(modeler,epoch,loss,accuracy,batch_idx)

            # if batch_idx % 4000 == 3999:
            #     save_model(modeler,epoch,loss,optimizer,batch_idx)  

            progress_bar.set_postfix({'loss': loss.item()})   

        else:
            break      
    
    
    accuracy = correct / total    
    total_loss = total_loss/iter 
    return total_loss,accuracy,optimizer


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=False)
        # for data in test_loader:
        for data in progress_bar:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = modeler(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            predicted = predicted.view(-1, 1)
            correct += (predicted == target).sum().item()
    acc = 1.0 * 100 * correct / total
    print('测试集准确率: %f%% [%d/%d]' % (acc, correct, total))
 
 
def save_model(model,epoch,total_loss,optimizer,batch_num):
    if is_transformer:
        name = './model_data/transformer2-model-' + str(epoch) + '-' + str(batch_num) + '-' +str(round(total_loss.item(),4) )
    else:

        name = './model_data/cnnmodel-' + str(epoch) + '-' + str(round(total_loss.item(),4) )
    
    checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': total_loss,
                    'batch_num':batch_num
                }
    torch.save(checkpoint, name)            

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    batch_num = checkpoint['batch_num']
    return model, optimizer, epoch, train_loss,batch_num


if __name__ == '__main__':



    is_gen_traindataset = False
    is_gen_testdataset = False
    is_load_model = True
    is_train = True
    path_to_model = "./model_data/transformer2-model-31-0-0.4141"
    start_epoch = -1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    conf = Configure()



    if is_gen_traindataset:
        train_dataset = PEDataSet(conf.train_path, True)

        with open('train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
    else:
        with open('train_dataset.pkl', 'rb') as f:
            train_dataset = pickle.load(f)  


    if is_gen_testdataset:
        test_dataset = PEDataSet(conf.test_path, False) 

        with open('test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
    else:
        with open('test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)  


    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)

    
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=2)


    # modeler = PEModel()
    modeler = TransformerModel()
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device))
    modeler.to(device)
 
    # optimizer = torch.optim.SGD(modeler.parameters(), lr=conf.lr,
    #                             weight_decay=conf.decay, momentum=conf.momentum)

    optimizer = torch.optim.Adam(modeler.parameters(), lr=conf.lr,weight_decay=conf.decay)                            
 
    print("========开始训练模型========")

    with open('training_log.txt', 'a') as file:
        # Write the epoch number and loss to the file
        file.write("============%s============\n" % datetime.now())

    start_batch = 0

    if is_load_model:
        modeler, optimizer, start_epoch, train_loss ,start_batch = load_model(modeler, path_to_model)

    modeler.to(device)


    if is_train:
        for i in range(start_epoch+1,conf.epochs):
            i_loss,i_acc,i_optimizer = train(i,start_batch)
            start_batch = 0
            # if i%1 == 0:
            save_model(modeler,i,i_loss,i_optimizer,0)

    print("========模型训练完成========")
    print("========开始测试模型========")
    test()
    
    print("========模型测试完成========")