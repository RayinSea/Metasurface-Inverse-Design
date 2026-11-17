import torch, anglespectrum
import numpy as np
from PredEnear import ExVector_broad1, PlotAmplPhase, PlotRealImag, ExArray_broad, FlipEx, en_mse
import matplotlib.pyplot as plt
from NetworkFiles.u_net import U_net
import os, scipy
from Combine_test import LargeMetaInverse

def FlipEx_reverse(m, n, N, data):
    if m <= (N - 1) / 2 and n >= (N - 1) / 2:
        data = np.fliplr(data)  # 逆操作：水平翻转
    elif m >= (N - 1) / 2 and n <= (N - 1) / 2:
        data = np.flipud(data)  # 逆操作：垂直翻转
    elif m > (N - 1) / 2 and n > (N - 1) / 2:
        data = np.fliplr(np.flipud(data))  # 逆操作：水平+垂直翻转

    return data

def ExArray_broad_reverse(N, ExComplexTemp2):
    # 计算 ExComplexTemp1 (N, N, 100)
    ExComplexTemp1 = np.zeros((N, N, 100), dtype=np.complex128)
    
    for m in range(N):
        for n in range(N):
            # 提取每个 10x10 子块并进行反转处理
            data = ExComplexTemp2[m*10:(m+1)*10, n*10:(n+1)*10]
            ExComplexTemp1[m, n, :] = np.reshape(FlipEx(m, n, N, data),(1, 100),order='F')
    
    # 生成 ExVector (N, N, 200)
    ExVector = np.zeros((N, N, 200), dtype=np.complex128)
    
    # 将 ExComplexTemp1 转换为 ExVector 的实部和虚部
    ExVector[:, :, 0:100] = np.real(ExComplexTemp1)
    ExVector[:, :, 100:200] = np.imag(ExComplexTemp1)
    
    return ExVector

def metalens_Ex(N, f, wavelength, p, coefficient):
    #um
    d = p #pixel size in real metalens

    center_x = np.linspace(-N*5+0.5, N*5-0.5, 10*N) * p
    center_y = center_x
    center_xx, center_yy = np.meshgrid(center_x, center_y)

    phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    phase = np.mod(phase, 2*np.pi) - np.pi

    # initial_Ex = np.load('Siemens.npy')
    # phase = np.angle(initial_Ex)

    ampl = np.zeros_like(phase) + 1

    
    E_real = ampl * np.cos(phase + 2 * np.pi * coefficient)
    E_imag = ampl * np.sin(phase + 2 * np.pi * coefficient)

    E = E_real + 1j*E_imag
    ExVector_reverse = ExArray_broad_reverse(N=N,ExComplexTemp2=E)

    return ExVector_reverse.transpose(2,0,1)

def S_meta_trad(N,f,wavelength,lambda_order,p=40e-3): 
    #um
    d = p #pixel size in real metalens
    
    center_x = np.linspace(-N*5+0.5, N*5-0.5, N) * p
    center_y = center_x
    center_xx, center_yy = np.meshgrid(center_x, center_y)

    phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    phase = np.mod(phase, 2*np.pi) - np.pi

    # initial_Ex = np.load('Siemens.npy')
    # phase = np.angle(initial_Ex)

    ampl = np.zeros_like(phase) + 0.8

    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file_path))

    filepath = os.path.join(parent_folder,'Modulation Capability','H500','PT.mat')
    data = scipy.io.loadmat(filepath)['PhaseTrans'] #（41，2，31）

    phase_list = data[:,0,lambda_order+1]
    R_list = np.linspace(25,125,41)

    S_metalens = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            closest_index = np.abs(phase_list-phase[i,j]).argmin()
            S_metalens[i,j] = R_list[closest_index].copy()

    return S_metalens

def MonoInverse(date, order, inverse, near, far, ifval, device='cpu'):

    N_list = [24,56,112,200,1000]; N = N_list[order]; f_list = [30,30,100,200,1000]
    wavelength = (400+(date-121001)*10)*1e-9
    amplmax = [2.8,25,45,80]
    ri_vmin_list = [-2.5, -1.5]
    ri_vmax_list = [2.5, 1.5]
    # 0. net model & ground truth data
    if N <= 200:
        model = U_net(200,1)
        model_path = os.path.join(f'./model data/{date}/N{N}/bestmodel.pth')
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
    # val
    if ifval:
        # data = np.load(f'./model data/{date}/N{N}/labels/val/labels_val_1.npy')
        data1 =  ExArray_broad_reverse(N, np.load(f'Ex_N24_GT_13.npy'))
        data = data1.transpose(2,0,1)

    # metalens & metasurface hologram example
    else:
        # initial_Ex = np.load('Siemens.npy')
        # temp = ExArray_broad_reverse(N=N, ExComplexTemp2=initial_Ex)
        # data = temp.transpose(2,0,1)
        
        data = metalens_Ex(N=N, 
                           f=f_list[order], 
                           wavelength=wavelength*1e6, 
                           p=40e-3, coefficient=0)

    Ex_input = torch.tensor(data,dtype=torch.float32).unsqueeze(0).to(device)

    # 1. inverse design
    if inverse:
        import time
        start = time.time()
        if N <= 200:
            S_inverse = model(Ex_input)
        elif N == 1000:
            S_inverse_comb = LargeMetaInverse(N=200,N_comb=5,date=date,data=data)
            S_inverse = torch.tensor(S_inverse_comb,dtype=torch.float32)

        print("total time: ", time.time()-start,"s")
        ExVectorPred = ExVector_broad1(N=N,S=S_inverse[0,0,:,:],vec_len=100,date=date,norm=0)
        # traditional design
    else:
        S_traditional = S_meta_trad(N=N,f=f_list[order],wavelength=wavelength*1e6,lambda_order=date-121001)
        ExVectorPred = ExVector_broad1(N=N,S=torch.tensor(S_traditional,dtype=torch.float32),
                                    vec_len=100,date=date,norm=1)
    
    # 2. Nearfield
    ExArrayPred = ExArray_broad(N=N,ExVector=ExVectorPred.cpu().detach().numpy(),vec_len=100)
    ExArrayGT = ExArray_broad(N=N,ExVector=Ex_input[0,:,:,:].permute(1,2,0).cpu().detach().numpy(),
                              vec_len=100)
    if near:
        # PlotRealImag(Ex_GT=ExArrayGT,Ex_Pred=ExArrayPred,combine_pixels=0,N=N,vmin_list=ri_vmin_list,vmax_list=ri_vmax_list)
        # PlotAmplPhase(Ex_GT=ExArrayGT,Ex_Pred=ExArrayPred,combine_pixels=0,N=N,amplmax=2.5)
        
        if ifval:
            # np.save(f'./model data/{date}/paper data/Exnear_N{N}_Pred.npy', ExArrayGT)
            # np.save(f'./model data/{date}/paper data/Exnear_N{N}_Inverse_Pred.npy', ExArrayPred)
            diff = en_mse(np.abs(ExArrayGT), np.abs(ExArrayPred))
            # np.save(f'./model data/{date}/paper data/enmsenear_N{N}_.npy', diff)
            print('near enmse: ', diff)

    # 3. Farfield
    if far:
        Efar_Pred = anglespectrum.AS(wavelength=wavelength, z=f_list[order]*1e-6, 
                                    P_map=np.angle(ExArrayPred), T_map=np.abs(ExArrayPred), dx=0.04e-6, dy=0.04e-6)
        Efar_GT = anglespectrum.AS(wavelength=wavelength, z=f_list[order]*1e-6, 
                                    P_map=np.angle(ExArrayGT), T_map=np.abs(ExArrayGT), dx=0.04e-6, dy=0.04e-6)
        # PlotAmplPhase(Ex_GT=Efar_GT,Ex_Pred=Efar_Pred,combine_pixels=0,N=N,amplmax=amplmax[order])
        if ifval:
            np.save(f'./model data/{date}/paper data/Exfar_N{N}_ForwardPred.npy', Efar_GT)
            np.save(f'./model data/{date}/paper data/Exfar_N{N}_InversePred.npy', Efar_Pred)
            diff = en_mse(np.abs(Efar_GT), np.abs(Efar_Pred))
            np.save(f'./model data/{date}/paper data/enmsefar_N{N}_.npy', diff)
            print('fresnel enmse: ', diff)

def MultiInverse(near, far, ifval, p):

    N = 24; f = 500; date_list = [121007, 121010, 121013]
    wavelength_list = [460e-9, 490e-9, 520e-9]
    ri_vmin_list = [-2.5, -1.5]
    ri_vmax_list = [2.5, 1.5]

    model = U_net(600,1)
    model_path = os.path.join(f'./model data/Multiwavelength/1210-07-10-13/N{N}/bestmodel.pth')
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    data_list = []
    # val
    if ifval:
        # for i, date in enumerate(date_list):
        #     data_list.append(np.load(f'./model data/{date}/N{N}/labels/val/labels_val_1.npy'))
        data_list.append(np.load('Ex_N24_GTinput_07.npy'))
        data_list.append(np.load('Ex_N24_GTinput_10.npy'))
        data_list.append(np.load('Ex_N24_GTinput_13.npy'))

    # metalens example
    else:
        for i,wavelength in enumerate(wavelength_list):
            data_list.append(metalens_Ex(N=N, 
                                         f=f, 
                                         wavelength=wavelength*1e6, 
                                         p=p, coefficient=0))
        
    data = np.concatenate(data_list, axis=0)
    Ex_input = torch.tensor(data,dtype=torch.float32).unsqueeze(0).to(device)

    # inverse design
    S_inverse = model(Ex_input)
    Ex_date_list = []
    for date in date_list:
        Ex_date_mlpPred = ExVector_broad1(N=N,S=S_inverse[0,0,:,:],
                                    vec_len=100,date=date,norm=0) # (N,N,200)
        Ex_date_list.append(Ex_date_mlpPred)
    Ex_dates = torch.cat(Ex_date_list,dim=0) # (200*m,N,N)
    
    for i, ExVectorPred in enumerate(Ex_date_list):
        # Nearfield
        ExArrayPred = ExArray_broad(N=N,ExVector=ExVectorPred.cpu().detach().numpy(),vec_len=100)
        ExArrayGT = ExArray_broad(N=N,ExVector=np.transpose(data_list[i],(1,2,0)),vec_len=100)
        
        if near:
            # PlotRealImag(Ex_GT=ExArrayGT,Ex_Pred=ExArrayPred,combine_pixels=0,N=N,vmin_list=ri_vmin_list,vmax_list=ri_vmax_list)
            PlotAmplPhase(Ex_GT=ExArrayGT,Ex_Pred=ExArrayPred,combine_pixels=0,N=N,amplmax=2.5)

        # Farfield
        if far:
            Efar_Pred = anglespectrum.AS(wavelength=wavelength_list[i], z=f*1e-6, 
                                        P_map=np.angle(ExArrayPred), T_map=np.abs(ExArrayPred), dx=0.04e-6, dy=0.04e-6)
            Efar_GT = anglespectrum.AS(wavelength=wavelength_list[i], z=f*1e-6, 
                                        P_map=np.angle(ExArrayGT), T_map=np.abs(ExArrayGT), dx=0.04e-6, dy=0.04e-6)
            PlotAmplPhase(Ex_GT=Efar_GT,Ex_Pred=Efar_Pred,combine_pixels=0,N=N,amplmax=30)
        
        # plt.savefig(f'fig{i}.png')

if __name__ == '__main__':

    # N_list = [24,56,112,200,1000]
    # MultiInverse(near=1, far=0, ifval=1, p=40e-3)
    # device 默认是 cpu
    MonoInverse(date=121007, order=0, inverse=1, near=1, far=0, ifval=1)
    plt.show()
    print()