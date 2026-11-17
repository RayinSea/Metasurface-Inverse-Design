import os, torch, train_mlp, h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PredMetasurface.VecTransArray import S_order, FlipEx, loc_board, FlipSub
from scipy.interpolate import RectBivariateSpline
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.colors import LinearSegmentedColormap
from PredEnear import en_mse

midnight_navy = (20/255, 30/255, 75/255)
blue_1 = (10/255, 50/255, 100/255)
classic_blue = (15/255, 76/255, 129/255)
turquoise_blue = (69/255, 184/255, 172/255)
emerald = (0/255, 152/255, 116/255)
green_1 = (80/255, 160/255, 100/255)
greenry = (136/255, 176/255, 75/255)
illuminating = (245/255, 223/255, 77/255)
yellow_1 = (240/255, 180/255, 50/255)
tangerine_tango = (221/255, 65/255, 36/255)
fiery_red = (190/255, 50/255, 45/255)

# 创建自定义颜色映射
colors = [midnight_navy, blue_1, classic_blue, turquoise_blue, emerald, green_1, 
          greenry, illuminating, yellow_1, tangerine_tango, fiery_red]
cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

if __name__  == "__main__":
    date = 121007
    unit_cell_order = 64

    model = train_mlp.ExModulationNet1210lambda()
    model_path = os.path.join('model data', str(date), f'{unit_cell_order}_bestmodel.params')
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)            
    model.eval() # 在 cpu 上做的测试

    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file_path))
    val_filename=os.path.join(parent_folder,'Array FDTD Data','Net Data',f'{unit_cell_order}_validationset{date-8100}.mat')
    val_loader = DataLoader(train_mlp.TensorExArray(val_filename), batch_size=1, shuffle=False)

    val_order = 54 # 3/17
    for val_i, val_data in enumerate(val_loader, 0):
        if val_i == val_order:
            val_inputs, val_labels = val_data[0], val_data[1]
            val_outputs = model(val_inputs)
            break
        else:
            continue       
    
    val_labels = val_labels.cpu().detach().numpy()
    real_labels = val_labels[0,0:100].reshape(10,10,order='F')
    imag_labels = val_labels[0,100:200].reshape(10,10,order='F')
    ampl_labels = np.abs(real_labels + 1j*imag_labels)
    phase_labels = np.angle(real_labels + 1j*imag_labels)

    val_outputs = val_outputs.cpu().detach().numpy()
    real_outputs = val_outputs[0,0:100].reshape(10,10,order='F')
    imag_outputs = val_outputs[0,100:200].reshape(10,10,order='F')
    ampl_outputs = np.abs(real_outputs + 1j*imag_outputs)
    phase_outputs = np.angle(real_outputs + 1j*imag_outputs)

    print("Ampl Labels: max: ", np.max(ampl_labels))
    # fig1, axs1 = plt.subplots(1,4)
    # axs1[0].imshow(real_labels, cmap=cmap)
    # axs1[0].set_title('Real Labels')
    # axs1[1].imshow(imag_labels, cmap=cmap)
    # axs1[1].set_title('Imag Labels')
    # axs1[2].imshow(real_outputs, cmap=cmap, vmin = np.min(real_labels), vmax = np.max(real_labels))
    # axs1[2].set_title('Real Outputs')
    # axs1[3].imshow(imag_outputs, cmap=cmap, vmin = np.min(imag_labels), vmax = np.max(imag_labels))
    # axs1[3].set_title('Imag Outputs')

    fig2, axs2 = plt.subplots(1,4,figsize=(10,2))
    axs2[0].imshow(ampl_labels, cmap=cmap)
    # axs2[0].set_title('Ampl Labels')
    axs2[1].imshow(phase_labels, cmap=cmap, vmin=-np.pi, vmax=np.pi)
    # axs2[1].set_title('Phase Labels')
    axs2[2].imshow(ampl_outputs, cmap=cmap, vmin=0, vmax=np.max(ampl_labels))
    # axs2[2].set_title('Ampl Outputs')
    im = axs2[3].imshow(phase_outputs, cmap=cmap, vmin=-np.pi, vmax=np.pi)
    # axs2[3].set_title('Phase Outputs')
    cbar = fig2.colorbar(im,ax=axs2[3])
    cbar.set_ticks([])
    # axs[i][j].set_title(title_list[i][j])
    for i in range(4):
        axs2[i].set_xticklabels([])
        axs2[i].set_yticklabels([])
        axs2[i].axis('off')

    plt.tight_layout()
    plt.show()

    print("")