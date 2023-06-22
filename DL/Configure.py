import os
 
 
class Configure:

    # model_type = "cnn"  # 
    # model_type = "trans2"  # 普通的transformer，12输入
    # model_type = "trans3"  # 加了直方图和字节熵  524输入
    # model_type = "trans4"  # 对直方图和字节熵进行归一化  524输入
    # model_type = "trans5"  # 全部输入，2381输入。
    model_type = "trans6"  # 直方图和字节熵的正确归一化
    model_normal = "/home/lizijian/lincode/ember/model_data/transformer2-model-33-0-0.4103"

    
    # train_dataset_path = "/home/lizijian/lincode/ember/DL/train_dataset_histogram2.pkl"
    # test_dataset_path = "/home/lizijian/lincode/ember/DL/test_dataset_histogram2.pkl"

    # 



    train_dataset_path = "/home/lizijian/lincode/ember/DL/train_dataset_histogram.pkl"
    test_dataset_path = "/home/lizijian/lincode/ember/DL/test_dataset_histogram.pkl"

    # train_load_model_path_histogram = "/home/lizijian/lincode/ember/model_data/transformer4-model-5-0-0.2019"
    test_load_model_path_histogram = "/home/lizijian/lincode/ember/model_data/transformer4-model-2-0-0.2345"

    # train_load_model_path_histogram = "/home/lizijian/lincode/ember/model_data/transformer4-model-5-0-0.2019"
    # test_load_model_path_histogram = "/home/lizijian/lincode/ember/model_data/transformer3-model-11-0-0.3634"


    if(model_type == "trans5"):  # 2381输入
        train_dataset_path = "/home/lizijian/lincode/ember/DL/train_dataset3.pkl"
        test_dataset_path = "/home/lizijian/lincode/ember/DL/test_dataset3.pkl"

        train_load_model_path= "/home/lizijian/lincode/ember/model_data/transformer5-model-1-0-0.6693"

        test_load_model_path = ""

    
    if(model_type == "trans6"):  # 对字节直方图和字节熵进行正确归一化 524维
        train_dataset_path = "/home/lizijian/lincode/ember/DL/train_dataset4.pkl"
        test_dataset_path = "/home/lizijian/lincode/ember/DL/test_dataset4.pkl"

        train_load_model_path= ""

        test_load_model_path = ""




    base_path = "/home/lizijian/lincode/ember/data/ember2018"
    train_path = []
    for i in range(6):
        train_path.append(os.path.join(base_path, "train_features_" + str(i) + ".jsonl"))
    test_path = os.path.join(base_path, "test_features.jsonl")
 
    batch_size = 64
    epochs = 50
    lr = 0.001
    decay = 0.0001
    momentum = 0.9