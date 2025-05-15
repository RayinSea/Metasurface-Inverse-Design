import h5py, anglespectrum as anglespectrum
import numpy as np
import scipy.io,os,torch
import matplotlib.pyplot as plt
from PredEnear import ExVector_broad1,ExArray_broad,IntPo,unitmean

def Padding(Ampl,Phase,lambda_order,combine_pixels,layer=2): # 在 Ampl 和 Phase 周围填补 Slab 的电场响应
    
    i = 2*layer+1
    m = layer; n = layer+1
    [p, q] = Ampl.shape
    N = round(p/10)

    AmplIndex = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data','N24_Ampl_xy1_Slab.mat'))['ampl']
    PhaseIndex = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data','N24_Phase_xy1_Slab.mat'))['phase']
    Ampl_padding = np.zeros([p*i, q*i]) + AmplIndex[lambda_order]
    Phase_padding = np.zeros_like(Ampl_padding) + PhaseIndex[lambda_order]
    Ampl_padding[p*m:p*n, q*m:q*n] = Ampl.copy()
    Phase_padding[p*m:p*n, q*m:q*n] = Phase.copy()

    if combine_pixels:
        return unitmean((2*layer+1)*N,Ampl_padding), unitmean((2*layer+1)*N,Phase_padding)
    else:
        return Ampl_padding, Phase_padding

if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file_path))

    # metasurface structure data
    N = 24
    S = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'S_N{N}.mat'))['S'] 
    S = S.astype(np.int32)

    lambda_order = 3-1 #112907
    date_list = [121001,121007,121013]

    vec_len = 100
    ExVector = ExVector_broad1(N=N, S=torch.tensor(S,dtype=torch.float32),
                               vec_len=vec_len,date=date_list[lambda_order])
    if vec_len == 600:
        Ex_Pred = ExArray_broad(N=N,ExVector=ExVector.cpu().detach().numpy(),vec_len=vec_len)[:,:,lambda_order]
    elif vec_len == 100:
        Ex_Pred = ExArray_broad(N=N,ExVector=ExVector.cpu().detach().numpy(),vec_len=vec_len)

    Ampl_Pred, Phase_Pred = np.abs(Ex_Pred), np.angle(Ex_Pred)

    [p, q] = [N*10, N*10]
    # FDTD results

    # Monitor_xy
    # Ex_xy1 = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'Ex_xy1_N{N}.mat'))['Ex']
    # [m, n, h] = Ex_xy1.shape;
    # Real_xy1_IntPo = IntPo(m=n,n=m,p=p,q=q,data=np.rot90(np.real(Ex_xy1[:,:,lambda_order])))
    # Imag_xy1_IntPo = IntPo(m=n,n=m,p=p,q=q,data=np.rot90(np.imag(Ex_xy1[:,:,lambda_order])))
    # Ampl_xy1_GT = np.abs(Real_xy1_IntPo + 1j*Imag_xy1_IntPo)
    # Phase_xy1_GT = np.angle(Real_xy1_IntPo + 1j*Imag_xy1_IntPo)

    # Monitor Array
    data_temp = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'Ex_N{N}.mat'))['Ex']
    Ex_GT = data_temp[:,:,lambda_order*6]; del data_temp
    Ampl_xy1_GT = np.abs(Ex_GT)
    Phase_xy1_GT = np.angle(Ex_GT)

    # Padding & unitmean
    combine_pixels = 0
    if combine_pixels:
        dx = 0.4e-6
        [p, q] = [N, N]
    else:
        dx = 0.04e-6
        [p, q] = [N*10, N*10]
    Ampl_Pred_padding, Phase_Pred_padding = Padding(Ampl_Pred, Phase_Pred, lambda_order=lambda_order,combine_pixels=combine_pixels)
    Ampl_IntPo_padding, Phase_IntPo_padding = Padding(Ampl_xy1_GT, Phase_xy1_GT, lambda_order=lambda_order,combine_pixels=combine_pixels)
   

    # wavelength data
    wavelength = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data','lambda 400-700.mat'))['lambda']
    # anglespectrum
    Efar_Pred = anglespectrum.AS(wavelength=wavelength[lambda_order*6,0], z=10e-6, 
                                 P_map=Phase_Pred_padding, T_map=Ampl_Pred_padding, dx=dx, dy=dx)
    Efar_GT = anglespectrum.AS(wavelength=wavelength[lambda_order*6,0], z=10e-6, 
                                 P_map=Phase_IntPo_padding, T_map=Ampl_IntPo_padding, dx=dx, dy=dx)

    # plot
    fig1, axs1 = plt.subplots(2,3,figsize=(10,5))

    data1 = np.abs(Efar_Pred[p*2:p*3, q*2:q*3])
    im = axs1[0,0].imshow(data1,cmap='jet',vmin=0,vmax=2.5)
    cbar = fig1.colorbar(im, ax=axs1[0,0])
    axs1[0,0].set_title('Ampl(MLP+AS)')

    # data2 = np.abs(Real_xy2_IntPo+1j*Imag_xy2_IntPo)
    data2 = np.abs(Efar_GT[p*2:p*3, q*2:q*3])
    im = axs1[0,1].imshow(data2,cmap='jet',vmin=0,vmax=2.5)
    cbar = fig1.colorbar(im, ax=axs1[0,1])
    axs1[0,1].set_title('Ampl_xy2')
    Amplmax = np.max(data2)
    im = axs1[0,2].imshow(np.abs(data1-data2),cmap='jet',vmin=0,vmax=2.5)
    axs1[0,2].set_title('Diff')
    cbar = fig1.colorbar(im, ax=axs1[0,2])

    data1 = np.angle(Efar_Pred[p*2:p*3, q*2:q*3])
    im = axs1[1,0].imshow(data1, cmap='jet',vmin=-np.pi,vmax=np.pi)
    cbar = fig1.colorbar(im, ax=axs1[1,0])
    axs1[1,0].set_title('Phase(MLP+AS)')

    # data2 = np.angle(Real_xy2_IntPo+1j*Imag_xy2_IntPo)
    data2 = np.angle(Efar_GT[p*2:p*3, q*2:q*3])
    im = axs1[1,1].imshow(data2, cmap='jet',vmin=-np.pi,vmax=np.pi)
    cbar = fig1.colorbar(im, ax=axs1[1,1])
    axs1[1,1].set_title('Phase_xy2')
    im = axs1[1,2].imshow(np.abs(data1-data2),cmap='jet',vmin=0,vmax=np.pi)
    axs1[1,2].set_title('Diff')
    cbar = fig1.colorbar(im, ax=axs1[1,2])

    plt.tight_layout()
    plt.show()

    print("")




