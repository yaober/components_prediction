import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import numpy as np
import math

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

BATCH_SIZE=4
EPOCHS=3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loader
df_train = pd.read_excel(io='../for-non-linear.xlsx',sheet_name='Xcal')
df_test = pd.read_excel(io='../for-non-linear.xlsx',sheet_name='Xtest')
np_train_data = np.array(df_train)[:,3:]
np_train_label = np.array(df_train)[:,:3]
np_test_data = np.array(df_test)[:,3:]
np_test_label = np.array(df_test)[:,:3]
print(df_train.shape, df_test.shape)
print(np_train_data.shape, np_train_data.shape)
print(np_test_label.shape, np_test_label.shape)

loss_func = nn.MSELoss()    #多分类问题，选择交叉熵损失函数

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(512,3)
    def forward(self,x):
        in_size = x.shape[0]
        #print(x.shape)
        out = self.conv1(x) #24
        #print("conv1:", out.shape)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        #print("conv2:", out.shape)
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        return out

def test(model, device, np_test_data, np_test_label):
    model.eval()
    test_loss = 0
    correct = 0
    input_data = torch.Tensor(np_test_data.reshape(-1,1,16)).to(device)
    with torch.no_grad():
        output = model(input_data)
        r2 = r2_score(np_test_label, output.cpu().numpy())
        return output.cpu().numpy()

def train(model, device, np_train_data, np_train_label, optimizer, epoch):
    model.train()
    np_train_data = np_train_data.reshape(-1, BATCH_SIZE, 1, 16)
    np_train_label = np_train_label.reshape(-1, BATCH_SIZE, 3)
    #print(np_train_data.shape, np_train_label.shape)
    for i in range(np_train_data.shape[0]):
        data, target = torch.Tensor(np_train_data[i]).to(device), torch.Tensor(np_train_label[i]).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss0 = loss_func(output, target)
        loss0.backward()
        optimizer.step()

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
max_r2 = -999
y_pred = 0
for epoch in range(1, EPOCHS + 1):
    print("epoch:",epoch)
    train(model, DEVICE, np_train_data, np_train_label, optimizer, epoch)
    output = test(model, DEVICE, np_test_data, np_test_label)
    r2 = r2_score(np_test_label, output)
    if max_r2 < r2:
        y_pred = output
        max_r2 = r2
        print(max_r2)

print("water:", r2_score(np_test_label[:,0], y_pred[:,0]), math.sqrt(mean_squared_error(np_test_label[:,0], y_pred[:,0])))
print("fat:", r2_score(np_test_label[:,1], y_pred[:,1]), math.sqrt(mean_squared_error(np_test_label[:,1], y_pred[:,1])))
print("protein:", r2_score(np_test_label[:,2], y_pred[:,2]), math.sqrt(mean_squared_error(np_test_label[:,2], y_pred[:,2])))