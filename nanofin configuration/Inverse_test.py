import torch, anglespectrum as anglespectrum
import numpy as np
from PredEnear import ExVector_broad1, PlotAmplPhase, PlotRealImag, ExArray_broad, FlipEx, en_mse
import matplotlib.pyplot as plt
from NetworkFiles.u_net import U_net
import os, scipy
import numpy as np
from skimage.metrics import structural_similarity as ssim
from Combine_test import LargeMetaInverse

def norm(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))

def E_pattern(N,f,wavelength,p=40e-3):
    # Generate target amplitude (a simple '3' shape)
    target = np.zeros((N*10, N*10))
    # 0
    target[50:190, 50:70] = 1
    target[50:190, 170:190] = 1
    target[50:70, 70:170] = 1
    target[170:190, 70:170] = 1
    # 3
    # target[30:50, 50:190] = 1
    # target[115:135, 50:190] = 1
    # target[200:220, 50:190] = 1
    # target[30:220, 170:190] = 1
    # Create U_target (complex field at z=30e-6 m with zero phase)
    # 随机生成相位
    phase = np.zeros_like(target) + np.pi*1.75
    # 构建复电场：幅值 * exp(i * phase)
    U_target = target * np.exp(1j * phase)
    U_target = np.array(U_target, dtype=np.complex64)

    return U_target

def metasurface_Ex(N,f,wavelength,U_target,p=40e-3):
    
    # Compute spatial frequencies
    fx = np.fft.fftfreq(N*10, d=p)
    fy = np.fft.fftfreq(N*10, d=p)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Wavenumber
    k = 2 * np.pi / wavelength

    # Compute the transfer function for back-propagation
    sqrt_arg = 1 - (wavelength**2) * (FX**2 + FY**2)
    sqrt_arg = np.clip(sqrt_arg, 1e-10, None)  # Avoid negative values for sqrt
    transfer_function = np.exp(-1j * k * f * np.sqrt(sqrt_arg))

    # Apply low-pass filter to remove evanescent waves
    mask = (FX**2 + FY**2) <= 1/(wavelength**2)
    transfer_function *= mask

    # Back-propagate to get initial field
    U_target_fft = np.fft.fft2(U_target)
    U_initial_fft = U_target_fft * transfer_function
    U_initial = np.fft.ifft2(U_initial_fft)

    ampl = np.abs(U_initial); phase = np.angle(U_initial)
    # phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    # phase = np.mod(phase, 2*np.pi) - np.pi
    # ampl = np.zeros_like(phase, dtype=np.complex128) + 0.8

    E_real_1 = ampl * np.cos(phase)
    E_imag_1 = ampl * np.sin(phase)

    E_1 = E_real_1 + 1j*E_imag_1
    ExVector_reverse_1 = ExArray_broad_reverse(N=N, ExComplexTemp2=E_1)

    # 落后 pi/2 的相位
    E_real_2 = ampl * np.sin(phase)
    E_imag_2 = - ampl * np.cos(phase)

    E_2 = E_real_2 + 1j*E_imag_2
    ExVector_reverse_2 = ExArray_broad_reverse(N=N, ExComplexTemp2=E_2)

    ExVector_reverse = np.concatenate((ExVector_reverse_1[:,:,0:100],ExVector_reverse_2[:,:,0:100],
                                       ExVector_reverse_1[:,:,100:200],ExVector_reverse_2[:,:,100:200]),axis=2)

    return ExVector_reverse.transpose(2,0,1)

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

def metalens_Ex(N,f,wavelength,p=40e-3):
    #um
    d = p #pixel size in real metalens

    center_x = np.linspace(-N*5+0.5, N*5-0.5, 10*N) * p
    center_y = center_x
    center_xx, center_yy = np.meshgrid(center_x, center_y)

    # E_temp = np.load('number3.npy')
    # ampl = np.abs(E_temp); phase = np.angle(E_temp)
    phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    phase = np.mod(phase, 2*np.pi) - np.pi
    ampl = np.zeros_like(phase, dtype=np.complex128) + 0.8

    E_real_1 = ampl * np.cos(phase)
    E_imag_1 = ampl * np.sin(phase)

    E_1 = E_real_1 + 1j*E_imag_1
    ExVector_reverse_1 = ExArray_broad_reverse(N=N, ExComplexTemp2=E_1)

    # 落后 pi/2 的相位
    E_real_2 = ampl * np.sin(phase)
    E_imag_2 = - ampl * np.cos(phase)

    E_2 = E_real_2 + 1j*E_imag_2
    ExVector_reverse_2 = ExArray_broad_reverse(N=N, ExComplexTemp2=E_2)

    ExVector_reverse = np.concatenate((ExVector_reverse_1[:,:,0:100],ExVector_reverse_2[:,:,0:100],
                                       ExVector_reverse_1[:,:,100:200],ExVector_reverse_2[:,:,100:200]),axis=2)

    return ExVector_reverse.transpose(2,0,1)

def S_meta_trad(N,f,wavelength,lambda_order,p=40e-3): 
    #um
    d = p #pixel size in real metalens
    
    center_x = np.linspace(-N*5+0.5, N*5-0.5, N) * p
    center_y = center_x
    center_xx, center_yy = np.meshgrid(center_x, center_y)

    phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    phase = np.mod(phase, 2*np.pi) - np.pi #[-pi,pi]
    
    # E_temp = np.load('number3.npy')
    # ampl = np.abs(E_temp); phase = np.angle(E_temp)

    S_metalens = (phase + np.pi) / (2 * np.pi) * 180

    return S_metalens

def MonoInverse(date, order, inverse, near, far, ifval=0, vec_len=200, device='cpu'):

    N_list = [24,56,112,200,1000]; N = N_list[order]; f_list = [30,30,100,200,1000]
    wavelength = 500e-9
    amplmax = [2.8,18,45,80]
    ri_vmin_list = [-2.5, -1.5]
    ri_vmax_list = [2.5, 1.5]

    if N <= 200:
        model = U_net(400,1)
        model_path = os.path.join(f'./model data/{date}/N{N}/bestmodel.pth')
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
    E_temp = E_pattern(N=N,
                       f=f_list[order],
                       wavelength=wavelength*1e6)
    
    # val
    if ifval:
        data = np.load(f'./model data/{date}/N{N}/labels/val/labels_val_1.npy')
        # data1 =  ExArray_broad_reverse(N, np.load(f'Ex_N24_GT.npy'))
        # data2 =  ExArray_broad_reverse(N, np.load(f'Ey_N24_GT.npy'))
        # data = np.concatenate((data1[:,:,0:100], data2[:,:,0:100],
        #                        data1[:,:,100:200], data2[:,:,100:200]), axis=2).transpose(2,0,1)

    # metalens example
    else:
        data = metasurface_Ex(N=N,
                           f=f_list[order], U_target=E_temp,
                           wavelength=wavelength*1e6)
        # data = metalens_Ex(N=N,
        #                    f=f_list[order],
        #                    wavelength=wavelength*1e6)
    
    Ex_input = torch.tensor(data,dtype=torch.float32).unsqueeze(0).to(device)

    # inverse design
    if inverse:
        import time
        start = time.time()
        if N <= 200:
            S_inverse = model(Ex_input)
        elif N == 1000:
            S_inverse_comb = LargeMetaInverse(N=200,N_comb=5,date=date,data=data)
            S_inverse = torch.tensor(S_inverse_comb,dtype=torch.float32)
            
        print("total time: ", time.time()-start,"s")
        ExVectorPred = ExVector_broad1(N=N, 
                                       S=S_inverse[0,0,:,:], 
                                       vec_len=vec_len, 
                                       date=date, norm=0)
    # traditional design
    else:
        
        S_traditional = S_meta_trad(N=N, 
                                    f=f_list[order], 
                                    wavelength=wavelength*1e6, 
                                    lambda_order=date-121001)
        
        ExVectorPred = ExVector_broad1(N=N, 
                                       S=torch.tensor(S_traditional,dtype=torch.float32), 
                                       vec_len=vec_len, 
                                       date=date, norm=1)
    
    # Nearfield
    ExArrayPred = ExArray_broad(N=N, 
                                ExVector=ExVectorPred.cpu().detach().numpy(), 
                                vec_len=vec_len)
    
    ExArrayGT = ExArray_broad(N=N, 
                              ExVector=Ex_input[0,:,:,:].permute(1,2,0).cpu().detach().numpy(), 
                              vec_len=vec_len)
    
    order_xy = 0
    if near:
        # PlotRealImag(Ex_GT=ExArrayGT[:,:,order_xy], 
        #              Ex_Pred=ExArrayPred[:,:,order_xy], 
        #              combine_pixels=0, 
        #              N=N, 
        #              vmin_list=ri_vmin_list, vmax_list=ri_vmax_list)
        
        # PlotAmplPhase(Ex_GT=ExArrayGT[:,:,order_xy], 
        #               Ex_Pred=ExArrayPred[:,:,order_xy], 
        #               combine_pixels=0, 
        #               N=N, 
        #               amplmax=2.8)

        if ifval:
            # np.save(f'./model data/{date}/paper data/Exnear_N{N}_Pred.npy', ExArrayGT[:,:,0])
            # np.save(f'./model data/{date}/paper data/Exnear_N{N}_Inverse_Pred.npy', ExArrayPred[:,:,0])
            diff_x = en_mse(np.abs(ExArrayGT[:,:,0]), np.abs(ExArrayPred[:,:,0]))
            # np.save(f'./model data/{date}/paper data/enmsenear_N{N}_x.npy', diff_x)
            print('near enmse_x: ', diff_x)
            # np.save(f'./model data/{date}/paper data/Eynear_N{N}_Pred.npy', ExArrayGT[:,:,1])
            # np.save(f'./model data/{date}/paper data/Eynear_N{N}_Inverse_Pred.npy', ExArrayPred[:,:,1])
            diff_y = en_mse(np.abs(ExArrayGT[:,:,1]), np.abs(ExArrayPred[:,:,1]))
            # np.save(f'./model data/{date}/paper data/enmsenear_N{N}_y.npy', diff_y)
            print('near enmse_y: ', diff_y)

    # Farfield
    if far:
        Exfar_Pred = anglespectrum.AS(wavelength=wavelength,  
                                     z=f_list[order]*1e-6, 
                                    P_map=np.angle(ExArrayPred[:,:,0]), 
                                    T_map=np.abs(ExArrayPred[:,:,0]), 
                                    dx=0.04e-6, dy=0.04e-6)
        
        Exfar_GT = anglespectrum.AS(wavelength=wavelength, 
                                   z=f_list[order]*1e-6, 
                                   P_map=np.angle(ExArrayGT[:,:,0]), 
                                   T_map=np.abs(ExArrayGT[:,:,0]), 
                                   dx=0.04e-6, dy=0.04e-6)
        
        Eyfar_Pred = anglespectrum.AS(wavelength=wavelength,  
                                     z=f_list[order]*1e-6, 
                                    P_map=np.angle(ExArrayPred[:,:,1]), 
                                    T_map=np.abs(ExArrayPred[:,:,1]), 
                                    dx=0.04e-6, dy=0.04e-6)
        
        Eyfar_GT = anglespectrum.AS(wavelength=wavelength, 
                                   z=f_list[order]*1e-6, 
                                   P_map=np.angle(ExArrayGT[:,:,1]), 
                                   T_map=np.abs(ExArrayGT[:,:,1]), 
                                   dx=0.04e-6, dy=0.04e-6)
        
        # hologram plot
        # gt = norm(np.square(np.abs(Exfar_GT))+np.square(np.abs(Eyfar_GT)))
        # inverse = norm(np.square(np.abs(Exfar_Pred))+np.square(np.abs(Eyfar_Pred)))
        # print('inverse design SSIM:', ssim(inverse, gt, data_range=1.0))

        # fig, axs = plt.subplots(1,2,figsize=(6,3))
        # axs[0].imshow(inverse);axs[1].imshow(gt)
                
        PlotAmplPhase(Ex_GT=Eyfar_GT, 
                      Ex_Pred=Eyfar_Pred, 
                      combine_pixels=0, 
                      N=N, 
                      amplmax=amplmax[order])


    if near or far:
        plt.show()

if __name__ == '__main__':

    # N_list = [24,56,112,200,1000]
    # device 默认是 cpu
    MonoInverse(date=250123, order=4, inverse=1, near=1, far=0, ifval=1)
    # MultiInverse(near=1, far=1, ifval=0)
    
    print() 