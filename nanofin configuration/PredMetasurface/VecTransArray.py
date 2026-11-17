import numpy as np
import torch
import matplotlib.pyplot as plt

def Unitcell_order(S_sub_zeros, unit_cell_order_flipmatrix, layer = 5):
    
    loop = True
    for i in range(0, layer+1):
        for j in range(0, layer+1):
            if S_sub_zeros[i, j] > 0:
                loop = False
                break
        if not loop:
            break
    return unit_cell_order_flipmatrix[i, j]


def loc_board(unit_cell_order_temp, i, j, N, layer = 5):

    shift_sign = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    if unit_cell_order_temp == (layer + 1) ** 2:
        [i_shift, j_shift] = [0, 0]
    else:
        i_shift = layer - (unit_cell_order_temp - 1) // (layer + 1)
        j_shift = layer + 1 - (unit_cell_order_temp - (layer - i_shift) * (layer + 1))
        if i < (N - 1) / 2 and j < (N - 1) / 2:  # 左上
            [i_shift, j_shift] = np.array([i_shift, j_shift]) * shift_sign[0]
        elif i <= (N - 1) / 2 and j >= (N - 1) / 2:  # 右上
            [i_shift, j_shift] = np.array([i_shift, j_shift]) * shift_sign[1]
        elif i >= (N - 1) / 2 and j <= (N - 1) / 2:  # 左下
            [i_shift, j_shift] = np.array([i_shift, j_shift]) * shift_sign[2]
        elif i > (N - 1) / 2 and j > (N - 1) / 2:  # 右下
            [i_shift, j_shift] = np.array([i_shift, j_shift]) * shift_sign[3]
    # 未翻转
    m = i + int(i_shift) - layer
    p = j + int(j_shift) - layer
    
    return m, p

def mask(i, j, m, p, N, layer = 5):
    
    rows = np.arange(m,m + 2*layer + 1)
    cols = np.arange(p,p + 2*layer + 1)

    if i <= (N - 1) / 2 and j >= (N - 1) / 2:  # 右上
        rows_n = rows.copy(); cols_n = np.flip(cols)
    elif i >= (N - 1) / 2 and j <= (N - 1) / 2:  # 左下
        rows_n = np.flip(rows); cols_n = cols.copy()
    elif i > (N - 1) / 2 and j > (N - 1) / 2:  # 右下
        rows_n = np.flip(rows); cols_n = np.flip(cols)
    else:
        rows_n = rows.copy(); cols_n = cols.copy()

    # 225 masks
    mask_list = np.zeros(((2*layer+1)**2, N, N))
    k = 0
    for jj in range(2*layer + 1):
        for ii in range(2*layer + 1):
            mask_list[k,rows_n[ii],cols_n[jj]] = 1
            k += 1

    return mask_list


def S_order(N, layer = 5):
    
    # 将 metasurface 周围结构5um厚的空白层
    S_expand = np.zeros((N + 2*layer, N + 2*layer))
    S_expand[layer:N + layer, layer:N + layer] = 1
    # mark metasurface 每个位置对应的 unit_cell_order
    unit_cell_order_flipmatrix = np.flip(np.flip(np.arange(1, (layer+1)**2+1).reshape(layer+1, layer+1), axis=0), axis=1)
    Ref = np.zeros([N, N])
    # [i, j] 对应的就是 S_expand 中的坐标位置
    for i in range(layer, N + layer):
        for j in range(layer, N + layer):
            # 5-10 和 63-68
            if i < 2*layer + 1 and j < 2*layer + 1:  # 左上
                S_sub_zeros = S_expand[i - layer:i + layer + 1, j - layer:j + layer + 1]
                Ref[i - layer, j - layer] = Unitcell_order(S_sub_zeros, unit_cell_order_flipmatrix)
            elif i < 2*layer + 1 and j > N - 2:  # 右上
                # 这不是我们需要的 11*11
                S_sub_zeros = np.flip(S_expand[i - layer:i + layer + 1, j - layer:j + layer + 1], axis=1)
                Ref[i - layer, j - layer] = Unitcell_order(S_sub_zeros, unit_cell_order_flipmatrix)
            elif i > N - 2 and j < 2*layer + 1:  # 左下
                S_sub_zeros = np.flip(S_expand[i - layer:i + layer + 1, j - layer:j + layer + 1], axis=0)
                Ref[i - layer, j - layer] = Unitcell_order(S_sub_zeros, unit_cell_order_flipmatrix)
            elif i > N - 2 and j > N - 2:  # 右下
                S_sub_zeros = np.flip(np.flip(S_expand[i - layer:i + layer + 1, j - layer:j + layer + 1], axis=1), axis=0)
                Ref[i - layer, j - layer] = Unitcell_order(S_sub_zeros, unit_cell_order_flipmatrix)
            else:
                S_sub_zeros = S_expand[i - layer:i + layer + 1, j - layer:j + layer + 1]
                Ref[i - layer, j - layer] = (layer + 1) ** 2

    for i in range(0, layer):
        Ref[i, layer + 1: N - layer - 1] = Ref[i, layer];
        Ref[N - 1 - i, layer + 1: N - layer - 1] = Ref[N - 1 - i, layer];
        Ref[layer: N - layer, i] = Ref[layer, i];
        Ref[layer: N - layer, N - 1 - i] = Ref[layer, N - 1 - i];

    return Ref # (N,N)


def FlipSub(i, j, N, Sub):

    if i <= (N - 1) / 2 and j >= (N - 1) / 2:  # 右上
        part = 2; Sub_flip = Sub
    elif i >= (N - 1) / 2 and j <= (N - 1) / 2:  # 左下
        part = 2
        Sub_flip = torch.flip(torch.flip(Sub, [0]), [1])
    elif i > (N - 1) / 2 and j > (N - 1) / 2:  # 右下
        part = 1
        Sub_flip = torch.flip(torch.flip(Sub, [1]), [0])
    else: # 左上
        part = 1; Sub_flip = Sub
    
    return part, Sub_flip

def FlipEx(m, n, N, data):

    if m <= (N - 1) / 2 and n >= (N - 1) / 2: # 右上
        data = data
    elif m >= (N - 1) / 2 and n <= (N - 1) / 2: # 左下
        data = np.flipud(np.fliplr(data))
    elif m > (N - 1) / 2 and n > (N - 1) / 2: # 右下
        data = np.fliplr(np.flipud(data))

    return data # (10,10)

# def FlipEx(m,n,N,data):
#     if m <= (N - 1) / 2 and n >= (N - 1) / 2:
#         data = torch.flip(data, [1])
#     elif m >= (N - 1) / 2 and n <= (N - 1) / 2:
#         data = torch.flip(data, [0])
#     elif m > (N - 1) / 2 and n > (N - 1) / 2:
#         data = torch.flip(torch.flip(data, [0]), [1]) 

#     return data # (10,10)

# def FlipSub(i,j,N,Sub):

#     if i <= (N - 1) / 2 and j >= (N - 1) / 2:  # 右上
#         Sub_flip = np.flip(Sub, axis=1)
#     elif i >= (N - 1) / 2 and j <= (N - 1) / 2:  # 左下
#         Sub_flip = np.flip(Sub, axis=0)
#     elif i > (N - 1) / 2 and j > (N - 1) / 2:  # 右下
#         Sub_flip = np.flip(np.flip(Sub, axis=1), axis=0)
#     else:
#         Sub_flip = Sub
    
#     return Sub_flip

if __name__ == '__main__':

    a = S_order(15)
    S = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13],
                  [2,3,4,5,6,7,8,9,10,11,12,13,14],
                  [3,4,5,6,7,8,9,10,11,12,13,14,15],
                  [4,5,6,7,8,9,10,11,12,13,14,15,16],
                  [5,6,7,8,9,10,11,12,13,14,15,16,17],
                  [6,7,8,9,10,11,12,13,14,15,16,17,18],
                  [7,8,9,10,11,12,13,14,15,16,17,18,19],
                  [8,9,10,11,12,13,14,15,16,17,18,19,20],
                  [9,10,11,12,13,14,15,16,17,18,19,20,21],
                  [10,11,12,13,14,15,16,17,18,19,20,21,22],
                  [11,12,13,14,15,16,17,18,19,20,21,22,23],
                  [12,13,14,15,16,17,18,19,20,21,22,23,24],
                  [13,14,15,16,17,18,19,20,21,22,23,24,25]])

    print("")