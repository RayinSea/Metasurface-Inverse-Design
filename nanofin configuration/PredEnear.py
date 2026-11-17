import os, torch, train_mlp, h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PredMetasurface.VecTransArray import S_order, FlipEx, loc_board, FlipSub
from scipy.interpolate import RectBivariateSpline

from matplotlib.colors import LinearSegmentedColormap

fix_value = 0
# 定义颜色（RGB值）
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

# colors = [fiery_red, tangerine_tango, yellow_1, illuminating, greenry, 
#           green_1, emerald, turquoise_blue, classic_blue, blue_1, midnight_navy]

# colors = [midnight_navy, classic_blue, turquoise_blue, emerald, 
#           greenry, illuminating, tangerine_tango, fiery_red]

cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

def IntPo(m, n, p, q, data): # (m,n) 是初始维度 (p,q) 是插值后的维度

    x = np.linspace(0, m-1, m)
    y = np.linspace(0, n-1, n)
    x_new = np.linspace(0, m-1, p)
    y_new = np.linspace(0, n-1, q)

    f = RectBivariateSpline(x, y, data)
    data_IntPo = f(x_new, y_new)
    return data_IntPo

# tensor 版本
def SarraytoStensorlist(S_input,norm=1):
    S_input_121 = S_input.t().flatten().unsqueeze(0)
    if norm == 1:
        S_input_norm = (S_input_121 - 1)/(180 - 1)
        return S_input_norm
    else:
        return S_input_121

def ExVector_broad1(N, S, vec_len, date, norm = 1, layer = 4):

    # load 所有模型
    model_list = []
    for part in range(1,3):
        for i in range(1, (layer+1)**2+1): # (1,26)
            model = train_mlp.ExModulationNet1()
            model_path = os.path.join('model data', str(date), f'part{part}', f'{i}_bestmodel.params')
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict)            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            model_list.append(model)

    # 训练时应该也需要将变量转移至 cpu 上
    S = S.to('cpu')
    # 判断每个 unitcell 对应的模型序号
    S_order_fix = S_order(N)
    ExWhole = torch.zeros([N, N, vec_len*2], dtype=torch.float32) # (N,N,400)

    for i in range(0, N):
        for j in range(0, N):
            # 划定 (2*layer+1)*(2*layer+1) 的 Sub 范围
            m,p = loc_board(S_order_fix[i, j], i, j, N)
            # 翻转/归一化/向量化
            part, S_input = FlipSub(i, j, N, S[m:m+2*layer+1, p:p+2*layer+1])
            S_input_temp = SarraytoStensorlist(S_input,norm=norm)

            model_temp = model_list[int(S_order_fix[i, j] - 1 + (part-1)*(layer+1)**2)]
            R_temp = model_temp(S_input_temp) # (1,400)
            ExWhole[i, j, :] = R_temp.clone()

    return ExWhole

def ExArray_broad(N, ExVector, vec_len):

    # 该函数不再训练时调用
    # ExVector 应为 numpy
    ExComplex = ExVector[:,:,0:vec_len] + 1j*ExVector[:,:,vec_len:vec_len*2]
    ExComplexTemp2 = np.zeros((N*10,N*10,2), dtype=np.complex128) # (N*10,N*10,2)
    for Exy_order in range(2):
        for m in range(N):
            for n in range(N):       
                data = np.reshape(ExComplex[m, n, Exy_order*100:(Exy_order+1)*100], 
                                  (10, 10), order='F') #(10,10)
                ExComplexTemp2[m*10:(m+1)*10, n*10:(n+1)*10, Exy_order] = FlipEx(m,n,N,data)

    return ExComplexTemp2 # (N*10,N*10,2)

def unitmean(N,data):

    new_data = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            new_data[i,j] = np.mean(data[i*10:(i+1)*10,j*10:(j+1)*10])

    return new_data

def PlotAmplPhase(Ex_Pred,Ex_GT,combine_pixels,N,amplmax=2.5):

    rows, cols = [2, 4]
    ampl_Pred = np.abs(Ex_Pred)
    phase_Pred = np.angle(Ex_Pred)

    ampl_IntPo = np.abs(Ex_GT)
    phase_IntPo = np.angle(Ex_GT)

    weights = ampl_IntPo/np.max(ampl_IntPo)

    if combine_pixels:
        ampl_Pred = unitmean(N=N,data=ampl_Pred)
        ampl_IntPo = unitmean(N=N,data=ampl_IntPo)
        phase_Pred = unitmean(N=N,data=phase_Pred)
        phase_IntPo = unitmean(N=N,data=phase_IntPo)
        weights = unitmean(N=N,data=weights)

    data_list = [[ampl_Pred, ampl_IntPo, np.abs(ampl_Pred-ampl_IntPo), np.abs(ampl_Pred-ampl_IntPo)*weights],
                 [phase_Pred, phase_IntPo, np.abs(np.abs(phase_Pred)-np.abs(phase_IntPo)), np.abs(np.abs(phase_Pred)-np.abs(phase_IntPo))*weights]]
    title_list = [['Ampl Pred','Ampl GT','Ampl Diff','Ampl Diff*Weights'],
                  ['Phase Pred','Phase GT','Phase Diff','Phase Diff*Weights']]
    vmin_list = [0, -np.pi]
    vmax_list = [amplmax, np.pi]
    # Plot
    fig, axs = plt.subplots(rows,cols,figsize=(13,5))
    for i in range(rows):
        for j in range(cols):
            im = axs[i,j].imshow(data_list[i][j], cmap=cmap,vmin=vmin_list[i],vmax=vmax_list[i]); 
            fig.colorbar(im,ax=axs[i][j])
            axs[i][j].set_title(title_list[i][j])
    
    plt.tight_layout()

def PlotRealImag(Ex_Pred,Ex_GT,combine_pixels,N,vmin_list = [-2.43,-2.73],vmax_list = [2.09,2.01]):

    rows, cols = [2, 4]
    # 懒得改了ampl是real，phase是imag
    ampl_Pred = np.real(Ex_Pred)
    phase_Pred = np.imag(Ex_Pred)

    ampl_IntPo = np.real(Ex_GT)
    phase_IntPo = np.imag(Ex_GT)

    weights = ampl_IntPo/np.max(ampl_IntPo)

    if combine_pixels:
        ampl_Pred = unitmean(N=N,data=ampl_Pred)
        ampl_IntPo = unitmean(N=N,data=ampl_IntPo)
        phase_Pred = unitmean(N=N,data=phase_Pred)
        phase_IntPo = unitmean(N=N,data=phase_IntPo)
        weights = unitmean(N=N,data=weights)

    data_list = [[ampl_Pred, ampl_IntPo, np.abs(ampl_Pred-ampl_IntPo), np.abs(ampl_Pred-ampl_IntPo)*weights],
                 [phase_Pred, phase_IntPo, np.abs(phase_Pred-phase_IntPo), np.abs(phase_Pred-phase_IntPo)*weights]]
    title_list = [['Real Pred','Real GT','Real Diff','Real Diff*Weights'],
                  ['Imag Pred','Imag GT','Imag Diff','Imag Diff*Weights']]

    # Plot
    fig, axs = plt.subplots(rows,cols,figsize=(13,5))
    for i in range(rows):
        for j in range(cols):
            im = axs[i,j].imshow(data_list[i][j], cmap=cmap,vmin=vmin_list[i],vmax=vmax_list[i]); 
            fig.colorbar(im,ax=axs[i][j])
            axs[i][j].set_title(title_list[i][j])
    
    plt.tight_layout()

def en_mse(ampl_true, ampl_pred):

    mse = np.mean((ampl_true - ampl_pred)**2)
    energy = np.mean(ampl_true**2)  # 或使用总能量 np.sum(ampl_true**2)
    return (mse / energy)

if __name__ == "__main__":

    import time
    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file_path))

    N = 1000; 
    S = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'S_N{N}.mat'))['S']
    # S = np.fliplr(S)
    S = S.astype(np.int32)
    # S = np.random.randint(1, 180, size=(N, N)); S = S.astype(np.int32)

    start = time.time()
    vec_len = 200
    ExVector = ExVector_broad1(N=N,
                               S=torch.tensor(S,dtype=torch.float32),
                               vec_len=vec_len,
                               date=250123) # (N,N,400) Real and Imag
    
    print("total time: ", time.time()-start,"s")
    Ex_Pred = ExArray_broad(N=N,
                            ExVector=ExVector.cpu().detach().numpy(),
                            vec_len=vec_len) # (N*10,N*10,2) Real and Imag

    print("")
    # Monitor Array
    data_temp = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'Exy_N{N}.mat'))['Exy']
    Ex_GT = data_temp; del data_temp

    # PlotRealImag(Ex_Pred=Ex_Pred[:,:,0],
    #              Ex_GT=Ex_GT[:,:,0],
    #              combine_pixels=0, N=N)
    
    PlotAmplPhase(Ex_Pred=Ex_Pred[:,:,0],
                  Ex_GT=Ex_GT[:,:,0],
                  combine_pixels=0,N=N,
                  amplmax=2.8)
    plt.show()
