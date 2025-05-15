from NetworkFiles.u_net import U_net
# from utils import Loss
import torch.nn.functional as F
import numpy as np

import torch, time, os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from PredEnear import ExVector_broad1

class MetasurfaceDataset(Dataset):
    def __init__(self,train,N,date):

        super(MetasurfaceDataset, self).__init__()
        # 获取基础路径
        self.date = date
        self.N = N
        self.flag = 'train' if train else 'val'
        self.root = f'./model data/{self.date}/N{self.N}'
        # 获取所需的路径
        self.img_name = [i for i in os.listdir(os.path.join(self.root,f'labels',self.flag))] # 'labels_train_21.png'
        self.img_list = [os.path.join(self.root,f'labels',self.flag, i) for i in self.img_name]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # 打开图像，转为灰度图
        # print(idx)
        inputs = np.load(self.img_list[idx])
        return torch.tensor(inputs,dtype=torch.float32)
    
def main(date=121007):
    # 设置基本参数信息
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # N200/N56
    batch_size = 1
    # N24
    # batch_size = 10
    num_epochs = 10
    N = 24

    train_dataset = MetasurfaceDataset(train=True, N=N, date=date)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = MetasurfaceDataset(train=False, N=N, date=date)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
    model = U_net(in_channels=200, out_channels=1) # 输入通道数，输出通道数
    model.to(device)

    # 加载预训练模型
    # model_path = os.path.join(f'./model data/{date}/N{N}/bestmodel.pth')
    # state_dict = torch.load(model_path, weights_only=True)
    # model.load_state_dict(state_dict)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad] # 定义需要优化的参数

    optimizer = optim.Adam(params=params, lr=0.005) # 0.005
    # N56/N200
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
    # N24
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 开始训练
    model.train()
    train_loss = np.zeros([num_epochs,1])
    val_loss = np.zeros([num_epochs, 1])
    val_loss_best = 1000
    
    i = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,train_data in enumerate(train_loader):
            Ex_list = []
            inputs = train_data.to(device)
            S_inverse = model(inputs)

            for batch_order in range(batch_size):
                Ex_mlpPred = ExVector_broad1(N=N,S=S_inverse[batch_order,0,:,:],
                                             vec_len=100,date=date,norm=0) # (N,N,200)
                Ex_list.append(Ex_mlpPred.permute(2,0,1))
            Ex_mlpPreds = torch.stack(Ex_list, dim=0) 
            loss = F.mse_loss(Ex_mlpPreds.to(device), inputs)

            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1} batch {i + 1}, training loss: {running_loss / (i+1):.8f}')

        scheduler.step()

        print(f'Epoch {epoch + 1}, training loss: {running_loss / (i+1):.8f}')
        train_loss[epoch] = running_loss / (i+1)
        
        # 在验证集上评估模型
        model.eval()
        val_loss_epoch = 0.0; val_i = 0
        with torch.no_grad():
            for val_i, val_data in enumerate(val_loader, 0):
                val_Ex_list = []
                val_inputs = val_data.to(device)
                val_S_inverse = model(val_inputs)

                for batch_order in range(batch_size):
                    val_Ex_mlpPred = ExVector_broad1(N=N,S=val_S_inverse[batch_order,0,:,:],
                                                     vec_len=100,date=date,norm=0) # (N,N,200)
                    val_Ex_list.append(val_Ex_mlpPred.permute(2,0,1))
                val_Ex_mlpPreds = torch.stack(val_Ex_list, dim=0)
                val_running_loss = F.mse_loss(val_Ex_mlpPreds.to(device), val_inputs)
                val_loss_epoch += val_running_loss.item()
 
        print(f'Epoch {epoch + 1}, val loss: {val_loss_epoch/(val_i+1):.8f}')
        val_loss[epoch] = val_loss_epoch / (val_i+1)

        # 更新 best model
        if val_loss[epoch] < val_loss_best:
            best_epoch = epoch+1
            val_loss_best = val_loss[epoch].copy()
            print('current best epoch: ', best_epoch)
            torch.save(model.state_dict(), f'./model data/{date}/N{N}/bestmodel.pth')

        np.save(f'./model data/{date}/N{N}/bestepoch.npy',best_epoch)
        np.save(f'./model data/{date}/N{N}/trainingloss.npy', train_loss)
        np.save(f'./model data/{date}/N{N}/validationloss.npy', val_loss)

    print('Training Completed')


if __name__ == '__main__':
    main()

