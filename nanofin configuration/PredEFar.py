import h5py, anglespectrum as anglespectrum
import numpy as np
import scipy.io,os,torch
import matplotlib.pyplot as plt
from PredEnear import ExVector_broad1,ExArray_broad,IntPo,unitmean

def Padding(Ampl,Phase,combine_pixels,layer=2): # 在 Ampl 和 Phase 周围填补 Slab 的电场响应
    
    i = 2*layer+1
    m = layer; n = layer+1
    [p, q] = Ampl.shape
    N = round(p/10)

    Exy_center = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data','Exy_N24_Slab_centermean.mat'))['Exy']
    AmplIndex = np.abs(Exy_center)
    PhaseIndex = np.angle(Exy_center)
    Ampl_padding = np.zeros([p*i, q*i]) + AmplIndex[0]
    Phase_padding = np.zeros_like(Ampl_padding) + PhaseIndex[0]
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

    vec_len = 200
    ExVector = ExVector_broad1(N=N, S=torch.tensor(S,dtype=torch.float32),
                               vec_len=vec_len,date=250123) # (N,N,400) Real and Imag
    Ex_Pred = ExArray_broad(N=N,ExVector=ExVector.cpu().detach().numpy(),vec_len=vec_len)

    Ampl_Pred, Phase_Pred = np.abs(Ex_Pred), np.angle(Ex_Pred)

    [p, q] = [N*10, N*10]
    # FDTD results
    # Monitor Array
    data_temp = scipy.io.loadmat(os.path.join(parent_folder,'Metasurface Data',f'Exy_N{N}.mat'))['Exy']
    Ex_GT = data_temp; del data_temp

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

    Ampl_Pred_padding, Phase_Pred_padding = Padding(Ampl_Pred[:,:,0], Phase_Pred[:,:,0], combine_pixels=combine_pixels)
    Ampl_IntPo_padding, Phase_IntPo_padding = Padding(Ampl_xy1_GT[:,:,0], Phase_xy1_GT[:,:,0], combine_pixels=combine_pixels)

    # anglespectrum
    Efar_Pred = anglespectrum.AS(wavelength=500e-9, z=10e-6, 
                                 P_map=Phase_Pred_padding, T_map=Ampl_Pred_padding, dx=dx, dy=dx)
    Efar_GT = anglespectrum.AS(wavelength=500e-9, z=10e-6, 
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




