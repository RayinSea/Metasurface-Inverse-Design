import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os
from scipy.io import savemat, loadmat
import torch.nn as nn
import torch.nn.functional as F

import sys
current_file_path = os.path.abspath(__file__)
main_folder = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_folder)  # 添加上级目录到模块搜索路径中
import train_mlp

if __name__ == "__main__":

    parent_folder = os.path.dirname(main_folder)

    date = 250123
    data_order = 0
    epoch_based = 0
    unit_cell_order = 1

    type = 'phase'
    R_cos_list = []
    R_sin_list = []
    cos_list = []
    sin_list = []

    data = loadmat(os.path.join(parent_folder, 'Array FDTD Data', 'Net data', f'{unit_cell_order}_trainingset{date}.mat'))

    input = data['input']
    Ex_raw = data['Ex']

    S_121_raw = input[data_order, :]
    S_array = S_121_raw.reshape((9, 9), order='F')
    S_array = 180 - np.fliplr(S_array)
    S_121 = (S_array.reshape((81,1),order='F')-1)/(180-1)
    # S_121 = (S_121_raw - 1)/(180-1)
    S_input_121_copy = S_121.copy()
    S_input_tensor = torch.tensor(S_input_121_copy, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_input = S_input_tensor.to(device)

    model = train_mlp.ExModulationNet1()
    len_list = 200
    order_list = np.array([0,1])
    combine_pixels = 1
    
    model.load_state_dict((torch.load(os.path.join(main_folder,'model data',f'{date}',f'{unit_cell_order}_bestmodel.params'))))
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 执行模型前向传播，得到预测结果
        R_output = model(S_input.squeeze())
        R_output_np = R_output.cpu().detach().numpy()

    Ex_real_pred = R_output_np[0:len_list]
    Ex_imag_pred = R_output_np[len_list:len_list*2]
    Ex_pred = Ex_real_pred + 1j*Ex_imag_pred
    Ex_real_GT = Ex_raw[data_order, 0:len_list]
    Ex_imag_GT = Ex_raw[data_order, len_list:len_list*2]
    Ex_GT = Ex_real_GT + 1j*Ex_imag_GT

    num = round(10/combine_pixels) 

    fig1, axes1 = plt.subplots(len(order_list)+1, 4)

    for i, lambda_order in enumerate(order_list):
        temp1 = Ex_GT[lambda_order * num**2: (lambda_order + 1) * num**2].reshape((num, num), order='F')
        ampl_GT = np.abs(temp1); phase_GT = np.abs(np.angle(temp1))

        temp2 = Ex_pred[lambda_order * num**2: (lambda_order + 1) * num**2].reshape((num, num), order='F')
        ampl_pred = np.abs(temp2); phase_pred = np.abs(np.angle(temp2))

        im1 = axes1[i,0].imshow(ampl_GT,cmap='jet',aspect='equal',vmin=0,vmax=2.5)
        fig1.colorbar(im1, ax=axes1[i,0])
        im2 = axes1[i,1].imshow(ampl_pred,cmap='jet',aspect='equal',vmin=0,vmax=2.5)
        fig1.colorbar(im2, ax=axes1[i,1])
        # im3 = axes1[i, 2].imshow(np.abs(cos_pred-cos_ground), cmap='jet', aspect='equal')
        im1 = axes1[i,2].imshow(np.abs(phase_GT),cmap='jet',aspect='equal',vmin=0,vmax=np.pi)
        fig1.colorbar(im1, ax=axes1[i,2])
        im2 = axes1[i,3].imshow(np.abs(phase_pred),cmap='jet',aspect='equal',vmin=0,vmax=np.pi)
        fig1.colorbar(im1, ax=axes1[i,3])

    # plt.tight_layout()
    plt.show()
    print("")