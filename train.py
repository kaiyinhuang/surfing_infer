import numpy as np
import json
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import copy

from torch.optim import AdamW
from torch.utils.data import DataLoader

kNumberEpoch = 40
kBatchSize = 16
kNumFeat = 3
kNumClass = 2
kTrainRatio = 0.5
kEvalRatio = 0.3
kTestRatio = 0.2

class full_connected(nn.Module):
    def __init__(self):
        super(full_connected, self).__init__()
        self.fc1 = nn.Linear(kNumFeat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3= nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, kNumClass)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x

class Zetao_Dataset(torch.utils.data.Dataset):
    def __init__(self, feats:np.ndarray, labels:np.ndarray):
        self.feats = feats
        self.labels = labels

    def __getitem__(self, idx) -> dict:
        item = dict()
        item['feat'] = self.feats[idx]
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train(feat_train:np.ndarray, feat_eval:np.ndarray, label_train:np.ndarray, label_eval:np.ndarray):
    # 定义模型
    fc_model = full_connected()

    # 定义优化器
    optim = AdamW(fc_model.parameters(), lr=5e-5)

    best_model = None
    accuracy_list = []
    best_accuracy = 0.0

    for epoch in range(kNumberEpoch):
        train_dataset = Zetao_Dataset(feats=feat_train, labels=label_train)
        train_loader = DataLoader(train_dataset, batch_size=kBatchSize, shuffle=True)

        eval_dataset = Zetao_Dataset(feats=feat_eval, labels=label_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=kBatchSize, shuffle=True)

        fc_model.train()
        with tqdm.tqdm(train_loader) as tq:
            for step, batch in enumerate(tq):
                # 从训练样本提取 feature 和 label
                feature = batch['feat']
                label = batch['label']
                # label = label - 1

                one_hot_label = F.one_hot(label, num_classes=2).float()

                # 前向传播
                predictions = fc_model(x=feature)
                
                # loss 计算
                loss = F.mse_loss(predictions, one_hot_label)
                
                # 清空优化器梯度
                optim.zero_grad()

                # 梯度计算
                loss.backward()

                # 更新模型
                optim.step()
    
        fc_model.eval()
        all_predictions = []
        all_labels = []
        with tqdm.tqdm(eval_loader) as tq, torch.no_grad():
            for step, batch in enumerate(tq):
                # 从训练样本提取 feature 和 label
                feature = batch['feat']
                label = batch['label']
                # label = label - 1

                # 前向传播
                predictions = fc_model(x=feature)
                _, predictions_indices = predictions.topk(1)
                predictions_indices = predictions_indices.T[0]

                all_predictions.append(predictions_indices)
                all_labels.append(label)

            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)

            accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
            accuracy_list.append(accuracy)

            if(accuracy > best_accuracy):
                best_accuracy = accuracy
                best_model = copy.deepcopy(fc_model)

            print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))

    print(f"accuracy curve: {accuracy_list}")
    return best_model

def test(model, feats:np.ndarray, labels:np.ndarray):
    model.eval()
    all_predictions = []
    all_labels = []

    test_dataset = Zetao_Dataset(feats=feats, labels=labels)
    test_loader = DataLoader(test_dataset, batch_size=kBatchSize, shuffle=True)

    with tqdm.tqdm(test_loader) as tq, torch.no_grad():
        for step, batch in enumerate(tq):
            # 从训练样本提取 feature 和 label
            feature = batch['feat']
            label = batch['label']
            label = label - 1

            # 前向传播
            predictions = model(x=feature)
            _, predictions_indices = predictions.topk(1)
            predictions_indices = predictions_indices.T[0]

            all_predictions.append(predictions_indices)
            all_labels.append(label)

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
        print(f'Test Accuracy {accuracy}')

def data_load():
    with open('dataset.json', 'r') as f:
        data_json = json.load(f)

        feature_1 = list()
        feature_2 = list()
        feature_3 = list()
        feature_4 = list()
        feature_5 = list()
        label = list()

        for i in range(0, len(data_json)):
            feature_1.append(data_json[i]["offshore_wind"])
            feature_2.append(data_json[i]["tides"])
            feature_3.append(data_json[i]["air_tempreature"])
            label.append(data_json[i]["suitable"])
        
        feature_1_ndarray = np.asarray(feature_1, dtype=np.float32)
        feature_2_ndarray = np.asarray(feature_2, dtype=np.float32)
        feature_3_ndarray = np.asarray(feature_3, dtype=np.float32)
        label_ndarray = np.asarray(label, dtype=np.int64)

        feature_matrix = np.stack(
            (feature_1_ndarray.T, feature_2_ndarray.T, feature_3_ndarray.T), 
            axis=0
        ).T

        indiceTrainRows = torch.tensor([int(i) for i in range(int(len(data_json)*kTrainRatio))]).long()
        indiceEvalRows = torch.tensor([int(i) for i in range(int(len(data_json)*kEvalRatio))]).long()
        indiceTestRows = torch.tensor([int(i) for i in range(int(len(data_json)*kTestRatio))]).long()

        feat_train = feature_matrix[indiceTrainRows]
        feat_eval = feature_matrix[indiceEvalRows]
        feat_test = feature_matrix[indiceTestRows]

        label_train = label_ndarray[indiceTrainRows]
        label_eval = label_ndarray[indiceEvalRows]
        label_test = label_ndarray[indiceTestRows]

        return feat_train, feat_eval, feat_test, label_train, label_eval, label_test

def main():
    # step 1: 读取数据
    feat_train, feat_eval, feat_test, label_train, label_eval, label_test = data_load()

    # step 2: 训练
    model = train(feat_train=feat_train, feat_eval=feat_eval, label_train=label_train, label_eval=label_eval)

    # step 3: 测试
    test(model=model, feats=feat_test, labels=label_test)
    

if __name__ == '__main__':
    main()