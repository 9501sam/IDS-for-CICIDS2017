#!/usr/bin/env python
# coding: utf-8

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam # Adam is just like SGD but faster
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import seaborn as sns  # Import Seaborn for the heatmap
import threading
import concurrent.futures
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import os
from collections import namedtuple
from sklearn.metrics import classification_report
import copy
import flwr as fl
from flwr.common import Metrics
from IPython.display import display, HTML
# display(HTML("<style>.container { width:80% !important; }</style>"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# # Helper Functions

def print_df_with_all_rows(df):
    pd.set_option('display.max_rows', None)
    display(df)
    pd.reset_option('display.max_rows')

def print_df_with_all_cols(df):
    pd.set_option('display.max_columns', None)
    display(df)
    pd.reset_option('display.max_columns')
    
def print_all_df(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    display(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

def summary_a_df(df):
    # Summary statistics for numerical columns
    numerical_summary = df.describe()

    # Summary statistics for string columns
    string_summary = df.describe(include=['object'])

    # Combine the summaries
    summary = pd.concat([numerical_summary, string_summary], axis=1)

    # Display the summary in table-like format
    display(summary.style.set_caption("Summary Statistics"))
    
def remove_nan_and_inf(df):
    df = df.dropna(how='any', axis=0, inplace=False)
    inf_condition = (df == np.inf).any(axis=1)
    df = df[~inf_condition]
    return df

def count_labels(df):
    label_counts = df[' Label'].value_counts()

    # Convert the value counts into a DataFrame for better formatting
    label_counts_df = pd.DataFrame(label_counts)

    # Rename the column to make it more descriptive
    label_counts_df.columns = ['Count']

    # Print the table
    display(label_counts_df)
    
def label_coding(df, label):
    label_encoder = LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    return df


# # Make csv File Training and Test set

# Define a named tuple to represent the return type
TrainTestSplit = namedtuple('TrainTestSplit', 
                            ['X_train', 'X_test', 'y_train', 'y_test', 'categories_as_list'])
def get_train_and_test(dfs):
    data = pd.concat(dfs, ignore_index=True)
    data = remove_nan_and_inf(data)
    
    # summary_a_df(data)
    
    # drop some unused feature
    data = data.drop(columns='Flow ID')

    # do label encoding to some feature
    label_encoder = LabelEncoder()
    data[" Source IP"] = label_encoder.fit_transform(data[" Source IP"])
    data[" Source Port"] = label_encoder.fit_transform(data[" Source Port"])
    data[" Destination IP"] = label_encoder.fit_transform(data[" Destination IP"])
    data[" Destination Port"] = label_encoder.fit_transform(data[" Destination Port"])
    data[" Timestamp"] = label_encoder.fit_transform(data[" Timestamp"])

    # Shuffle all the rows of the DataFrame
    # data = data.sample(frac=1).reset_index(drop=True)
    # summary_a_df(data)
    
    ### get `X` from `data`
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    X = data.drop(columns=' Label')
    headers_list = X.columns.tolist()
    headers_to_standard = [header for header in headers_list if header not in 
               [' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp']]
    ct = ColumnTransformer([
        ('somename', StandardScaler(), headers_to_standard)
    ], remainder='passthrough')
    X = ct.fit_transform(X)
    
    ### get `y` from `data`
    y = data.iloc[:, -1:]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)
    categories = ohe.categories_
    categories_as_list = []
    for i, label in enumerate(categories[0]):
        categories_as_list.append(label)
        
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, shuffle=True)
    return TrainTestSplit(X_train, X_test, y_train, y_test, categories_as_list)

# Tue = get_train_and_test([df_Tue])


dataset_dir = './TrafficLabelling/'
def read_csv(csv_file):
    file_path = os.path.join(dataset_dir, csv_file)
    # Try different encodings until the file is successfully read
    df = None
    for encoding in ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break  # Stop trying encodings once the file is successfully read
        except UnicodeDecodeError:
            continue
    return df

def make_label_amount(df, label, amount):
    df = remove_nan_and_inf(df)
    # Filter the DataFrame to select rows with the specified label
    label_rows = df[df[' Label'] == label]
    # Take the first 'amount' rows with the specified label
    df_subset = label_rows.head(amount)
    df_remaining = df[df[' Label'] != label]
    # Concatenate the subset DataFrame with the remaining rows
    df_concatenated = pd.concat([df_remaining, df_subset], ignore_index=True)
    return df_concatenated

df_Mon  = read_csv("Monday-WorkingHours.pcap_ISCX.csv")

df_Tue  = read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
df_Tue = make_label_amount(df_Tue, "BENIGN", 9900)
count_labels(df_Tue)

df_Wed  = read_csv("Wednesday-workingHours.pcap_ISCX.csv")
df_Wed = make_label_amount(df_Wed, "BENIGN", 9900)
df_Wed = make_label_amount(df_Wed, "DoS Hulk", 9900)
df_Wed = make_label_amount(df_Wed, "DoS GoldenEye", 9900)
count_labels(df_Wed)

df_Thu1 = read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df_Thu2 = read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df_Thu = pd.concat([df_Thu1, df_Thu2], ignore_index=True)
df_Thu = make_label_amount(df_Thu, "BENIGN", 9900)
count_labels(df_Thu)

df_Fri1 = read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df_Fri2 = read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df_Fri3 = read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
df_Fri = pd.concat([df_Fri1, df_Fri2, df_Fri3], ignore_index=True)

df_Mon_and_Fri = pd.concat([df_Fri1, df_Fri2, df_Fri3], ignore_index=True)
### TODO
df_Mon_and_Fri = make_label_amount(df_Mon_and_Fri, "BENIGN", 9900)
df_Mon_and_Fri = make_label_amount(df_Mon_and_Fri, "DDoS", 9900)
df_Mon_and_Fri = make_label_amount(df_Mon_and_Fri, "PortScan", 9900)
count_labels(df_Mon_and_Fri)


# # Function for Training Models
def training_model(model, 
                   train_test,
                   learning_rate=0.0001, n_epochs=300, batch_size=512):
    # get data from train_test
    X_train            = train_test.X_train           
    X_test             = train_test.X_test            
    y_train            = train_test.y_train           
    y_test             = train_test.y_test            
    categories_as_list = train_test.categories_as_list
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # prepare model and training parameters
    batches_per_epoch = len(X_train) // batch_size

    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    # start training 
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        # with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        for i in range(batches_per_epoch):
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            # bar.set_postfix(
            #     loss=float(loss),
            #     acc=float(acc)
            # )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        # print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
        print({epoch}, end='')
        
    # display result
    # Restore best model
    model.load_state_dict(best_weights)

    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    # plt.savefig('cross_entropy.png')
    # plt.show()

    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    # plt.savefig('accuracy.png')
    # plt.show()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    y_pred_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_classes = torch.argmax(y_test, dim=1).cpu().numpy()

    from sklearn.metrics import confusion_matrix

    confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

    # Assuming you have already computed the confusion matrix
    confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

    # Define your custom class labels
    # class_labels = ['normal', 'tcpfin', 'tcppush', 'tcprst', 'tcpsyn', 'udpflood']
    class_labels = categories_as_list
    # class_labels = [i for i in range(1, 16)]
    # Create a heatmap of the confusion matrix with custom labels
    # categories_as_list = [item for sublist in categories for item in sublist]  # Flatten the categories list
    # class_labels = categories_as_list

    plt.figure(figsize=(16, 12))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')
    # plt.show()
    
    # Calculate the classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    display(report_df)

# Mon = get_train_and_test([df_Mon])
Tue = get_train_and_test([df_Tue])
Wed = get_train_and_test([df_Wed])
Thu = get_train_and_test([df_Thu])
Mon_and_Fri = get_train_and_test([df_Mon_and_Fri])


# # Defining models
class ANN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.hidden1 = nn.Linear(num_inputs, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, num_outputs)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x
# ann_Tue = ANN(Tue.X_test.size()[1], len(Tue.categories_as_list)).to(DEVICE)
# ann_Wed = ANN(Wed.X_test.size()[1], len(Wed.categories_as_list)).to(DEVICE)
# ann_Thu = ANN(Thu.X_test.size()[1], len(Thu.categories_as_list)).to(DEVICE)

class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.hidden1 = nn.Linear(num_inputs, 500)
        self.hidden2 = nn.Linear(500, 500)
        self.hidden3 = nn.Linear(500, 500)
        self.hidden4 = nn.Linear(500, 500)
        self.hidden5 = nn.Linear(500, 500)
        self.hidden6 = nn.Linear(500, 500)
        self.output = nn.Linear(500, num_outputs)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = self.output(x)
        return x
# dnn_Tue = DNN(Tue.X_test.size()[1], len(Tue.categories_as_list)).to(DEVICE)
# dnn_Wed = DNN(Wed.X_test.size()[1], len(Wed.categories_as_list)).to(DEVICE)
# dnn_Thu = DNN(Thu.X_test.size()[1], len(Thu.categories_as_list)).to(DEVICE)
dnn_Mon_and_Fri = DNN(Mon_and_Fri.X_test.size()[1], len(Mon_and_Fri.categories_as_list)).to(DEVICE)

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.hidden1 = nn.Linear(num_inputs, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.hidden3 = nn.Linear(50, 50)
        self.output = nn.Linear(50, num_outputs)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x
# mlp_Tue = MLP(Tue.X_test.size()[1], len(Tue.categories_as_list)).to(DEVICE)
# mlp_Wed = MLP(Wed.X_test.size()[1], len(Wed.categories_as_list)).to(DEVICE)
# mlp_Thu = MLP(Thu.X_test.size()[1], len(Thu.categories_as_list)).to(DEVICE)


# ### DNN
training_model(model=dnn_Mon_and_Fri, train_test=Mon_and_Fri)


# ### FlowerClient
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):#是知識載入
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)# 剛聚合完的權重
        #global test 對每global round剛聚合完的gobal model進行測試 要在Local_train之前測試
        # 通常第1 round測出來會是0
        # 在训练或测试结束后，保存模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/Before_local_train_model.pth")
        accuracy = test(net, testloader, start_IDS, client_str,f"global_test",True)
        print("accuracy",accuracy)
                    # 将总体准确率和其他信息写入 "accuracy-baseline.csv" 文件
        with open(f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/accuracy-gobal_model_{client_str}.csv", "a+") as file:
            # file.write(str(RecordAccuracy))
            # file.writelines("\n")
            # 添加标题行
            file.write(f"{client_str}_gobal_model_Accuracy\n")
            # 写入Accuracy数据
            file.write(str(accuracy) + "\n")

        train(net, trainloader, epochs=num_epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}#step1上傳給權重，#step2在server做聚合，step3往下傳給server

    def evaluate(self, parameters, config):
        # local test
        # 這邊的測試結果會受到local train的影響
        # 在训练或测试结束后，保存模型
        torch.save(net.state_dict(), f"./FL_AnalyseReportfolder/{today}/{client_str}/{Choose_method}/After_local_train_model.pth")
        accuracy = test(net, testloader, start_IDS, client_str,f"local_test",True)
        self.set_parameters(parameters)#更新現有的知識#step4 更新model
        return accuracy, len(testloader.dataset), {"accuracy": accuracy}


# ### Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:53388",
    client=FlowerClient(),
)
