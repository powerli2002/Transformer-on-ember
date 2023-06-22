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


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    batch_num = checkpoint['batch_num']
    return model, optimizer, epoch, train_loss


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=False)
        # for data in test_loader:
        for data in progress_bar:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            if is_transformer:
                inputs = inputs.unsqueeze(dim=1) 

            outputs = modeler(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            predicted = predicted.view(-1, 1)
            correct += (predicted == target).sum().item()
    acc = 1.0 * 100 * correct / total
    print('测试集准确率: %f%% [%d/%d]' % (acc, correct, total))
 
     

if __name__ == '__main__':
    conf = Configure()


    is_gen_traindataset = False
    is_gen_testdataset = False
    is_load_model = True
    is_train = False
    path_to_model = "./model_data/transformer2-model-30-0-0.4159"
    start_epoch = -1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    



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

    if is_load_model:
        modeler, optimizer, start_epoch, train_loss  = load_model(modeler, path_to_model)



    # if is_train:
    #     for i in range(start_epoch+1,conf.epochs):
    #         i_loss,i_acc,i_optimizer = train(i)
    #         if i%1 == 0:
    #             save_model(modeler,i,i_loss,i_optimizer,0)

    # print("========模型训练完成========")
    print("========开始测试模型========")
    test()
    
    print("========模型测试完成========")