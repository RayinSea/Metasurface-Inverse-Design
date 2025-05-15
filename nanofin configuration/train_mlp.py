import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io

n_parts = 1

class ExModulationNet1(nn.Module):

    def __init__(self):
        super(ExModulationNet1, self).__init__()
        self.fc1 = nn.Linear(81, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 400)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def TensorExArray(filename):
    # 加载 .mat 文件
    data = scipy.io.loadmat(filename)
    inputs = data['input']  # 假设 input 是 .mat 文件中的一个变量
    # 将 NumPy 数组转换为 PyTorch 张量
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    # 归一化 inputs
    inputs_min = 1
    inputs_max = 180   
    inputs_norm = (inputs_tensor - inputs_min) / (inputs_max - inputs_min)

    # phases_scaled = phases_tensor / torch.pi

    # 使用归一化后的数据创建 Dataset
    Ex = data['Ex']
    Ex_tensor = torch.tensor(Ex, dtype=torch.float32)
    dataset = TensorDataset(inputs_norm, Ex_tensor)
    
    return dataset

def trainMLP(date,train_filename, val_filename, num_epochs, epoch_based, dynamic_lr):

    train_loader = DataLoader(TensorExArray(train_filename), batch_size=128, shuffle=False)
    val_loader = DataLoader(TensorExArray(val_filename), batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 训练循环
    model = ExModulationNet1()
    model.to(device)
    model_best = ExModulationNet1()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    if dynamic_lr == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)

    train_loss = np.zeros([num_epochs,1])
    val_loss = np.zeros([num_epochs, 1])
    val_loss_best = 1000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, train_data in enumerate(train_loader, 0):
            # 获取输入和相位
            # 将数据移动到 GPU
            inputs, labels = train_data[0].to(device), train_data[1].to(device)
            # 梯度归零
            optimizer.zero_grad()
            # 前向传播、反向传播、优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 打印统计信息
            running_loss += loss.item()
        # 更新学习率
        if dynamic_lr == 1:
            scheduler.step()

        print(f'Epoch {epoch + 1}, training loss: {running_loss / (i+1):.8f}')
        train_loss[epoch] = running_loss / (i+1)
        
        # 在验证集上评估模型
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for val_i, val_data in enumerate(val_loader, 0):
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_inputs)
                val_loss_epoch += criterion(val_outputs, val_labels)

        print(f'Epoch {epoch + 1}, test loss: {val_loss_epoch/(val_i+1):.8f}')
        val_loss[epoch] = val_loss_epoch.cpu().detach().numpy()/(val_i+1)

        # 更新 best model
        if val_loss_epoch < val_loss_best:
            model_best.load_state_dict(model.state_dict())
            best_epoch = epoch+1
            val_loss_best = np.float64(val_loss_epoch).copy()

    date = date
    torch.save(model_best.state_dict(), f'./model data/{date}/part2/{unit_cell_order}_bestmodel.params')
    np.save(f'./model data/{date}/part2/{unit_cell_order}_bestepoch.npy',best_epoch)

    np.save(f'./model data/{date}/part2/{unit_cell_order}_trainingloss.npy', train_loss)
    np.save(f'./model data/{date}/part2/{unit_cell_order}_validationloss.npy', val_loss)
    print('Training Completed')

if __name__ == "__main__":
    import os
    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file_path))

    type= 250123
    for unit_cell_order in range(1, 37):
        print(f'unit_cell_order: {unit_cell_order}')
        trainMLP(date=type,train_filename=os.path.join(parent_folder,'Array FDTD Data','Net Data_part2',f'{unit_cell_order}_trainingset{type}.mat'), 
                val_filename=os.path.join(parent_folder,'Array FDTD Data','Net Data_part2',f'{unit_cell_order}_validationset{type}.mat'), 
                num_epochs=200, epoch_based=0, dynamic_lr=1)
