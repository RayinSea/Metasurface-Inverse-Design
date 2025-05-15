import numpy as np
import scipy.io

# wavelength, z, dx, dy 的单位均为 m
# wavelength 为单一数值，不是一系列波长的 np 数组
# P_map 是 n*n 的相位分布
# T_map 是 n*n 的振幅分布
# dx, dy 均使用了 unit cell 的周期长度 1e-6
# 函数返回的 n*n 数组是像面电场强度的实部分布，如需比较光强可以改为 E0^2

def AS(wavelength, z, P_map, T_map, dx, dy):
    M, N = P_map.shape
    k = 2 * np.pi / wavelength

    # 使用fftfreq生成正确的频率轴并移位
    fx = np.fft.fftshift(np.fft.fftfreq(M, dx))  # x方向频率（列，水平）
    fy = np.fft.fftshift(np.fft.fftfreq(N, dy))  # y方向频率（行，垂直）

    # 生成网格，indexing='ij'确保形状为(M, N)
    fxx, fyy = np.meshgrid(fx, fy, indexing='ij')

    # 计算传递函数H
    temp = 1 - (wavelength * fxx)**2 - (wavelength * fyy)**2
    H = np.zeros((M, N), dtype=complex)
    valid = temp > 0
    H[valid] = np.exp(1j * k * z * np.sqrt(temp[valid]))

    # 光场传播计算
    Ei = T_map * np.exp(1j * P_map)
    U1 = np.fft.fft2(Ei)
    U2 = np.fft.fftshift(U1) * H  # 应用传递函数
    U3 = np.fft.ifftshift(U2)
    Eo = np.fft.ifft2(U3)

    return Eo

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # metalens example
    p = 0.5e-6 #period
    t = 1e-6
    N = 100
    d = 0.1e-6 #pixel size in real metalens
    r = N*p+(p-0.1e-6)/2 
    wavelength = 0.65e-6
    f = 5e-4

    x = np.linspace(-r, r, int(p/d*(2*N+1)))
    y = x

    xx, yy = np.meshgrid(x, y)

    dis = np.sqrt(xx**2+yy**2)
    center_x = np.linspace(-N, N, 2*N+1) * p
    center_y = center_x
    center_xx, center_yy = np.meshgrid(center_x, center_y)

    Phase = -2*np.pi/wavelength*(np.sqrt(center_xx**2+center_yy**2+f**2)-f)
    Trans = np.zeros_like(Phase) + 1

    Enorm = AS(650e-9, f, Phase, Trans, p, p)

    plt.imshow(Enorm)
    plt.show()

