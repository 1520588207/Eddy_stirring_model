import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
#from line_profiler_pycharm import profile
from numba import njit, float64,int32,float32
from numba import cuda
import math
import sys
import torch
import numpy as np
import scipy.io
from scipy.io import loadmat
from numba import njit
import matplotlib.pyplot as plt


def sw_f(lat):


    # Constants
    DEG2RAD = np.pi / 180
    OMEGA = 7.292e-5  # s^-1 (A.E.Gill p.597)

    # Calculation
    f = 2 * OMEGA * np.sin(lat * DEG2RAD)

    return f



def repeat_temp(Temp, a):
    # 获取数组的形状
    rows, cols = Temp.shape

    # 创建一个空数组，用于存放重复后的结果
    repeated_array = np.zeros((rows, cols * a))

    # 使用 for 循环重复数组
    for i in range(a):
        # 将 Temp 的每一列复制到 repeated_array 的对应位置
        repeated_array[:, i * cols: (i + 1) * cols] = Temp

    return repeated_array

def mondip(temp_an_r):
    r = 1
    xp = np.linspace(1, 100, 65)
    yp = np.linspace(1, 100, 65)
    xxp, yyp = np.meshgrid(xp, yp)
    x = np.linspace(1, 100, 100)
    y = np.linspace(1, 100, 100)
    xx, yy = np.meshgrid(x, y)
    t_r = np.interp2(xx, yy, temp_an_r, xxp, yyp)

    SST_mean = t_r[:65, :65]
    d = np.zeros((65, 65))
    for i in range(len(xp)):
        for j in range(len(yp)):
            d[i, j] = np.sqrt((i - 33) ** 2 + (j - 33) ** 2)
    D = d.ravel()
    SST = SST_mean.ravel()
    I = np.argsort(D)
    D_sort = D[I]
    SST_sort = SST[I]
    C, ia, ic = np.unique(D_sort, return_index=True, return_inverse=True)
    mean_sst = np.zeros(len(ia))
    for i in range(len(ia) - 1):
        mean_sst[i] = np.mean(SST_sort[ia[i]:ia[i + 1]])
    mean_sst[-1] = SST_sort[ia[-1]]
    SST_r = np.zeros_like(SST_sort)
    SST_r[ic] = mean_sst
    SST_mon = np.reshape(SST_r, (len(xp), len(yp)))
    SST_dip = SST_mean - r * SST_mon
    return SST_mon, SST_dip, t_r



def meshgrid_numba(x, y):
    xx = np.zeros((len(y), len(x)))
    yy = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            xx[j, i] = x[i]
            yy[j, i] = y[j]
    return xx, yy




def regular_grid_interpolator(x, y, z, xi, yi):
    # 确保输入数据合法性
    assert len(x) == z.shape[1], "x 和 z 的维度不匹配"
    assert len(y) == z.shape[0], "y 和 z 的维度不匹配"

    # 计算 x 和 y 的步长
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    # 找到 xi 和 yi 在 x 和 y 上的索引
    ix = (xi - x[0]) / dx
    iy = (yi - y[0]) / dy

    # 计算插值结果
    zi = np.zeros((len(xi), len(yi)))
    for i in range(len(xi)):
        for j in range(len(yi)):
            ix_floor = int(np.floor(ix[i]))
            iy_floor = int(np.floor(iy[j]))
            ix_ceil = int(np.ceil(ix[i]))
            iy_ceil = int(np.ceil(iy[j]))
            if ix_floor < 0 or ix_ceil >= len(x) or iy_floor < 0 or iy_ceil >= len(y):
                continue
            zi[j, i] = ((x[ix_ceil] - xi[i]) * (y[iy_ceil] - yi[j]) * z[iy_floor, ix_floor] +
                        (xi[i] - x[ix_floor]) * (y[iy_ceil] - yi[j]) * z[iy_floor, ix_ceil] +
                        (x[ix_ceil] - xi[i]) * (yi[j] - y[iy_floor]) * z[iy_ceil, ix_floor] +
                        (xi[i] - x[ix_floor]) * (yi[j] - y[iy_floor]) * z[iy_ceil, ix_ceil]) / (dx * dy)
    return zi



#sig3=(int32, int32, float32[:, ::1], int32,
# int32, int32, int32, float32[:, ::1],float32[::1])
#@cuda.jit(sig3)
#def con2(i, j, next_data, hm, hn, hm2, hn2, h, B):

#    m1, n1 = next_data.shape

#    B[0]=float32(0)
#    for k in range(hm):
#        for l in range(hn):
#            ii = i + k - hm2
#            jj = j + l - hn2
#            if ii >= 0 and ii < m1 and jj >= 0 and jj < n1:
#                B[0] += next_data[ii, jj] * h[k, l]



@cuda.jit
def con2(i, j, next_data, hm, hn, hm2, hn2, h):

    m1, n1 = next_data.shape

    B = 0
    for k in range(hm):
        for l in range(hn):
            ii = i + k - hm2
            jj = j + l - hn2
            if ii >= 0 and ii < m1 and jj >= 0 and jj < n1:
                B += next_data[ii, jj] * h[k, l]

    return B

sig=(float64[:,:,::1], float64[:,:, ::1],float64[::1],float64[:,::1],float64[:,::1],float64[::1],float64[::1],float64[::1],float64,
                                                                                float64,float64,float64,float64,float64,float64,float64,float64)
@cuda.jit(sig)
def calculate_velocityo(buf_0, buf_1, At, x2, y2, t, CX, Rt, g, f, rho0s, U, V, dx, dy, X1, X2):
    idxWithinGridx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStridex = cuda.gridDim.x * cuda.blockDim.x

    idxWithinGridy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    gridStridey = cuda.gridDim.y * cuda.blockDim.y

    # Define shared memory for P1, P2, P3 calculations
    sP1 = cuda.shared.array(shape=(32, 32), dtype=float64)
    sP2 = cuda.shared.array(shape=(32, 32), dtype=float64)
    sP3 = cuda.shared.array(shape=(32, 32), dtype=float64)

    for m in range(200):
        for i in range(idxWithinGridx, buf_0.shape[0], gridStridex):
            for j in range(idxWithinGridy, buf_0.shape[1], gridStridey):

                # Calculate P1, P2, P3 using shared memory for optimization
                if cuda.threadIdx.x < 32 and cuda.threadIdx.y < 32:
                    sP1[cuda.threadIdx.x, cuda.threadIdx.y] = At[m] * (
                            1 - (((x2[i, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (
                                                                      t[m] <= t[40]) \
                                                             + At[m] * (
                                                                     1 - (((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (
                                                                             y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (
                                                                      (t[40] < t[m]) & (t[m] <= t[160])) \
                                                             + At[m] * (
                                                                     1 - (((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (
                                                                             y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (
                                                                      t[m] > t[160])

                    sP2[cuda.threadIdx.x, cuda.threadIdx.y] = At[m] * (
                            1 - (((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (
                                                                      t[m] <= t[40]) \
                                                             + At[m] * (
                                                                     1 - (((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (
                                                                             y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (
                                y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
                                                             + At[m] * (
                                                                     1 - (((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (
                                                                             y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (
                                y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

                    sP3[cuda.threadIdx.x, cuda.threadIdx.y] = At[m] * (
                            1 - (((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (
                                                                      t[m] <= t[40]) \
                                                             + At[m] * (
                                                                     1 - (((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (
                                                                             y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (
                                y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
                                                             + At[m] * (
                                                                     1 - (((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (
                                                                             y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(
                        -(((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (
                                y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

                cuda.syncthreads()

                # Perform calculations using shared memory
                if (i < buf_0.shape[0] - 1) and (j < buf_0.shape[1] - 1):
                    buf_0[i, j, m] = -1 / f / rho0s * ((sP2[cuda.threadIdx.x, cuda.threadIdx.y] - sP1[cuda.threadIdx.x, cuda.threadIdx.y]) / dy) + U
                    buf_1[i, j, m] = 1 / f / rho0s * ((sP3[cuda.threadIdx.x, cuda.threadIdx.y] - sP1[cuda.threadIdx.x, cuda.threadIdx.y]) / dx) + V

                # Ensure all threads have finished writing to buf_0 and buf_1 before continuing
                cuda.syncthreads()

                # Reset initial conditions for buf_0 and buf_1 at m = 0
                if m == 0:
                    buf_0[i, j, m] = 0
                    buf_1[i, j, m] = 0

sig=(float64[:,:,::1], float64[:,:, ::1],float64[::1],float64[:,::1],float64[:,::1],float64[::1],float64[::1],float64[::1],float64,
                                                                                float64,float64,float64,float64,float64,float64,float64,float64)
@cuda.jit(sig)
def calculate_velocity(buf_0, buf_1, At, x2, y2, t, CX, Rt, g, f, rho0s, U, V, dx, dy, X1, X2):


    idxWithinGridx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStridex = cuda.gridDim.x * cuda.blockDim.x

    idxWithinGridy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    gridStridey = cuda.gridDim.y * cuda.blockDim.y

#    xl=cuda.local.array((x2.shape[0],x2.shape[1]), dtype=np.float32)
#    yl=cuda.local.array((y2.shape[0],y2.shape[1]), dtype=np.float32)

#    xl=x2
#    yl=y2

    for m in range(200):

     for i in range(idxWithinGridx, buf_0.shape[0], gridStridex):
        for j in range(idxWithinGridy, buf_0.shape[1], gridStridey):
#       sA = cuda.shared.array(shape=(5,5), dtype=float32)
#       sB = cuda.shared.array(shape=(5, 5), dtype=float32)


           P1 = At[m] * (1 - (((x2[i, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + CX[0] * t[m]) / Rt[m])** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
            + At[m] * (1 - (((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
            + At[m] * (1 - (((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           P2 = At[m] * (1 - (((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
               + At[m] * (1 - (((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
               + At[m] * (1 - (((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2+ (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           P3 = At[m] * (1 - (((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40])\
                + At[m] * (1 - (((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
                + At[m] * (1 - (((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           buf_0[i, j, m] = -1 / f / rho0s * ((P2 - P1) / dy) + U
           buf_1[i, j, m] = 1 / f / rho0s * ((P3 - P1) / dx) + V


           if m == 0:
            buf_0[i, j, m]=0
            buf_1[i, j, m]=0



# Define the kernel signature
sig = (float32[:, :, ::1], float32[:, :, ::1], float32[::1], float32[:, ::1], float32[:, ::1], float32[::1],
       float32[::1], float32[::1], float32, float32, float32, float32, float32, float32,
       float32, float32, float32)

# 定义核函数签名
sig = (float32[:, :, ::1], float32[:, :, ::1], float32[::1], float32[:, ::1], float32[:, ::1],
       float32[::1], float32[::1], float32[::1], float32, float32, float32, float32,
       float32, float32, float32, float32, float32,int32[::1])

@cuda.jit(sig)
def calculate_velocity2(buf_0, buf_1, At, x2, y2, t, CX, Rt, g, f, rho0s, U, V, dx, dy, X1, X2, global_sync):
    i, j = cuda.grid(2)

    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    if i == 0 and j == 0:
        global_sync[0] = 0  # 初始化同步标志

    cuda.syncthreads()  # 确保所有线程块在初始化同步标志后再继续

    for m in range(200):
        P1 = At[m] * (1 - (((x2[i, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + CX[0] * t[m]) / Rt[m])** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
             + At[m] * (1 - (((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
             + At[m] * (1 - (((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

        P2 = At[m] * (1 - (((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
             + At[m] * (1 - (((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
             + At[m] * (1 - (((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2+ (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

        P3 = At[m] * (1 - (((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40])\
             + At[m] * (1 - (((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
             + At[m] * (1 - (((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

        buf_0[i, j, m] = -1 / f / rho0s * ((P2 - P1) / dy) + U
        buf_1[i, j, m] = 1 / f / rho0s * ((P3 - P1) / dx) + V

        if m == 0:
            buf_0[i, j, m] = 0
            buf_1[i, j, m] = 0

        # 全局同步
        if i == 0 and j == 0:
            cuda.atomic.add(global_sync, 0, 1)  # 当前线程块完成更新

    # 等待所有线程块完成更新
    cuda.syncthreads()

    # 检查全局同步标志，确保所有线程块都已完成更新
    if i == 0 and j == 0:
        while global_sync[0] < cuda.gridDim.x * cuda.gridDim.y:
            pass  # 等待其他线程块完成更新

    # 所有线程块同步完成后继续执行
    cuda.syncthreads()




sig=(float32[:,:,::1], float32[:,:, ::1],float32[::1],float32[:,::1],float32[:,::1],float32[::1],float32[::1],float32[::1],float32,
                                                                                float32,float32,float32,float32,float32,float32,float32,float32)
@cuda.jit(sig)
def calculate_velocity2_old(buf_0, buf_1, At, x2, y2, t, CX, Rt, g, f, rho0s, U, V, dx, dy, X1, X2):

    i, j = cuda.grid(2)




    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    grid = cuda.cg.this_grid()

    grid.sync()

    for m in range(200):

           P1 = At[m] * (1 - (((x2[i, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + CX[0] * t[m]) / Rt[m])** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
            + At[m] * (1 - (((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
            + At[m] * (1 - (((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           P2 = At[m] * (1 - (((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40]) \
               + At[m] * (1 - (((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
               + At[m] * (1 - (((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2+ (y2[i + 1, j] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i + 1, j] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i + 1, j] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           P3 = At[m] * (1 - (((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + CX[0] * t[m]) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] <= t[40])\
                + At[m] * (1 - (((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X1 + CX[m] * (t[m] - t[40]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * ((t[40] < t[m]) & (t[m] <= t[160])) \
                + At[m] * (1 - (((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * math.exp(-(((x2[i, j + 1] + (X2 + CX[160] * (t[m] - t[160]))) / Rt[m]) ** 2 + (y2[i, j + 1] / Rt[m]) ** 2) / 2) * rho0s * g * (t[m] > t[160])

           buf_0[i, j, m] = -1 / f / rho0s * ((P2 - P1) / dy) + U
           buf_1[i, j, m] = 1 / f / rho0s * ((P3 - P1) / dx) + V


           if m == 0:
            buf_0[i, j, m]=0
            buf_1[i, j, m]=0

    grid.sync()






def uv_cal_UV(rx, ry, sc, C, U, V, Rend, A, T, lat, rho0):
    a = rx
    b = ry

#    print('rx:',rx)
    t = np.linspace(0, T, 200) * 24 * 60 * 60
    Rt = (1 / 40 * np.arange(201) * (np.arange(201) <= 40) + 1 * ((np.arange(201) > 40) & (np.arange(201) <= 160)) +
          (-1 / 40) * (np.arange(201) - 200) * (np.arange(201) > 160)) * Rend
    At = (1 / 40 * np.arange(201) * (np.arange(201) <= 40) + 1 * ((np.arange(201) > 40) & (np.arange(201) <= 160)) +
          (-1 / 40) * (np.arange(201) - 200) * (np.arange(201) > 160)) * A
    xx = np.linspace(-sc, sc, a) * Rend
    dx = xx[1] - xx[0]
    yy = np.linspace(-sc, sc, b) * Rend
    dy = yy[1] - yy[0]

    [x2, y2] = meshgrid_numba(xx, yy)
    f = sw_f(lat)
    g = 9.8
    rho0s = rho0
    CX = C * np.ones(201)
    X1 = CX[0] * t[40]
    X2 = X1

    ug = np.zeros((a - 1, b - 1, 200), dtype=np.float32)
    vg = np.zeros((a - 1, b - 1, 200), dtype=np.float32)

    #    ug[:,:,0] = -1 / f / rho0 * ((P[1:, :-1] - P[:-1, :-1]) / dy) + U
    #    vg[:,:,0] = 1 / f / rho0 * ((P[:-1, 1:] - P[:-1, :-1]) / dx) + V

    buf_0 = cuda.to_device(ug)
    buf_1 = cuda.to_device(vg)

    threadsperblock = (12, 12)
    blockspergrid_x = math.ceil(ug.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ug.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    calculate_velocity[blockspergrid, threadsperblock](buf_0, buf_1, float32(At), float32(x2), float32(y2), float32(t), float32(CX), float32(Rt), float32(g), float32(f), float32(rho0), float32(U), float32(V), float32(dx),
                                                       float32(dy), float32(X1), float32(X2))

#    calculate_velocity[blockspergrid, threadsperblock](buf_0, buf_1, At, x2, y2, t, CX, Rt, g, f, rho0, U, V, dx,
#                                                       dy, X1, X2)
    At_np=np.float32(At)
    At_dev = cuda.to_device(At_np)

    x2_np = x2.astype(np.float32)
    x2_dev = cuda.to_device(x2_np)

    x2_dev = cuda.to_device(x2)
    y2_dev = cuda.to_device(y2)

    y2_np = x2.astype(np.float32)
    y2_dev = cuda.to_device(y2_np)

    t_np = t.astype(np.float32)
    t_dev = cuda.to_device(t_np)

    CX_np = CX.astype(np.float32)
    CX_dev = cuda.to_device(CX_np)

    Rt_np = Rt.astype(np.float32)
    Rt_dev = cuda.to_device(Rt_np)


    rho0s_np=rho0s.astype(np.float32)
    rho0s_dev = cuda.to_device(rho0s_np)


    g_np=np.float32(g)
    g_dev=cuda.to_device(g_np)


    f_np=f.astype(np.float32)
    f_dev=cuda.to_device(f_np)

    rho0_np=rho0.astype(np.float32)
    rho0_dev=cuda.to_device(rho0_np)


    U_np=U.astype(np.float32)
    U_dev=cuda.to_device(U_np)


    V_np=np.float32(V)
    V_dev=cuda.to_device(V_np)

    dx_np=dx.astype(np.float32)
    dx_dev=cuda.to_device(dx_np)

    dy_np=dy.astype(np.float32)
    dy_dev=cuda.to_device(dy_np)

    X1_np=dy.astype(np.float32)
    X1_dev=cuda.to_device(X1_np)

    X2_np=dy.astype(np.float32)
    X2_dev=cuda.to_device(X2_np)


#    calculate_velocity[blockspergrid, threadsperblock](buf_0, buf_1, At_dev, x2_dev, y2_dev, t_dev, CX_dev, Rt_dev, rho0s_np, g_np, f_np, rho0_np, U_np, 0, dx_np,
#                                                       dy_np, X1_np, X2_np)
#    calculate_velocity[blockspergrid, threadsperblock](buf_0, buf_1, At_np, x2_np, y2_np, t_np, CX_np, Rt_np, rho0s_np, g_np, f_np, rho0_np, U_np, 0, dx_np,
#                                                       dy_np, X1_np, X2_np)
#    sys.exit()


    ug = buf_0.copy_to_host()
    vg = buf_1.copy_to_host()

    return CX, ug, vg


@cuda.jit
def process_data(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h):
    idxWithinGridx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStridex = cuda.gridDim.x * cuda.blockDim.x

    idxWithinGridy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    gridStridey = cuda.gridDim.y * cuda.blockDim.y
#    plt.contourf(ug[:, :, 1])
#    plt.show()


#    sys.exit()

    b, a, d = TEMP.shape
#    i, j = cuda.grid(2)
#    grid = cuda.cg.this_grid()
    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2
#    d=2
    for m in range(d - 1):


        if (m % 2) == 0:
         data = buf_0
         next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

#        curr_temp = data[i, j]

#        print(next_data[5,5])

        for i in range(idxWithinGridx, data.shape[0], gridStridex):
            for j in range(idxWithinGridy, data.shape[1], gridStridey):

#               print('i:',i,'j:',j)
               if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
                     next_temp = (-vg[i, j, m] * (data[i, j] - TEMP[i - 1, j, m]) / dy[i,j] -
                     ug[i, j, m] * (data[i, j] - TEMP[i, j - 1, m]) / dx[i,j]) * dt[i,j] + (
                     aa * (TEMP[i + 1, j, m] + TEMP[i - 1, j, m] - 2 * data[i, j]) / dy[i,j] ** 2 +
                     bb * (TEMP[i, j + 1, m] + TEMP[i, j - 1, m] - 2 * data[i, j]) / dx[i,j] ** 2) * dt[i,j] + data[i, j]
               else:
                     next_temp = 0


               next_data[i, j] = next_temp
#               TEMP[i,j,m+1]=next_temp

        cuda.syncthreads()

#        grid.sync()


        for i in range(idxWithinGridx, buf_0.shape[0], gridStridex):
            for j in range(idxWithinGridy, buf_0.shape[1], gridStridey):

              if (i >= 0) & (i < (data.shape[0] - 1)) & (j >= 0) & (j < (data.shape[1] - 1)):
                TEMP[i, j, m + 1] = con2(i, j, next_data, hm, hn, hm2, hn2, h)
                next_data[i, j]=TEMP[i, j, m + 1]
              elif (i == data.shape[0]-1):
                TEMP[i,j,m+1] = TEMP[i,j,1]
                next_data[i, j] = TEMP[i,j,1]

              elif (j == data.shape[1]-1):
               TEMP[i,j,m+1] = TEMP[i,j,1]
               next_data[i, j] = TEMP[i, j, 1]


        cuda.syncthreads()

#        grid.sync()


@cuda.jit()
def process_data_normal(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h, tnext_data):
    b, a, d = TEMP.shape
    i, j = cuda.grid(2)

    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    grid = cuda.cg.this_grid()

    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2

    grid.sync()
    for m in range(d - 1):
        if (m % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            curr_temp = data[i, j]
            next_temp = (-vg[i, j, m] * (curr_temp - TEMP[i - 1, j, m]) / dy[i, j] -
                         ug[i, j, m] * (curr_temp - TEMP[i, j - 1, m]) / dx[i, j]) * dt[i, j] + (
                         aa * (TEMP[i + 1, j, m] + TEMP[i - 1, j, m] - 2 * curr_temp) / dy[i, j] ** 2 +
                         bb * (TEMP[i, j + 1, m] + TEMP[i, j - 1, m] - 2 * curr_temp) / dx[i, j] ** 2) * dt[i, j] + curr_temp
            next_data[i, j] = next_temp
            tnext_data[i, j]= next_temp
        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            next_temp = 0
            next_data[i, j] = next_temp
            tnext_data[i, j] = next_temp

        grid.sync()

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):

            m1, n1 = tnext_data.shape
            local_result = 0.0
            for k in range(hm):
                for l in range(hn):
                    ii = i + k - hm2
                    jj = j + l - hn2
                    if ii >= 0 and ii < m1 and jj >= 0 and jj < n1:
                        local_result += tnext_data[ii, jj] * h[k, l]
            TEMP[i, j, m + 1] = local_result
            next_data[i, j] = TEMP[i, j, m + 1]


        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            TEMP[i, j, m + 1] = TEMP[i, j, 1]
            next_data[i, j] = TEMP[i, j, 1]

        grid.sync()

 #       if i == 5 and j == 5:
 #           print("After second sync: m =", m)

@cuda.jit
def process_data_normal2(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h, tnext_data):
    b, a, d = TEMP.shape
    i, j = cuda.grid(2)

    # Define shared memory size for TEMP slice
    sm_size = cuda.shared.array(shape=(16, 16), dtype=float32)

    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    grid = cuda.cg.this_grid()

    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2

    grid.sync()

    for m in range(d - 1):
        if (m % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            curr_temp = data[i, j]
            next_temp = (-vg[i, j, m] * (curr_temp - TEMP[i - 1, j, m]) / dy[i, j] -
                         ug[i, j, m] * (curr_temp - TEMP[i, j - 1, m]) / dx[i, j]) * dt[i, j] + (
                            aa * (TEMP[i + 1, j, m] + TEMP[i - 1, j, m] - 2 * curr_temp) / dy[i, j] ** 2 +
                            bb * (TEMP[i, j + 1, m] + TEMP[i, j - 1, m] - 2 * curr_temp) / dx[i, j] ** 2) * dt[
                            i, j] + curr_temp
            next_data[i, j] = next_temp
            tnext_data[i, j] = next_temp
        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            next_temp = 0
            next_data[i, j] = next_temp
            tnext_data[i, j] = next_temp

        grid.sync()

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):

            m1, n1 = tnext_data.shape
            local_result = 0.0

            # Load data into shared memory
            sm_size[i % 16, j % 16] = tnext_data[i, j]

            # Synchronize threads to ensure all data is loaded
            cuda.syncthreads()

            for k in range(hm):
                for l in range(hn):
                    ii = i + k - hm2
                    jj = j + l - hn2
                    if ii >= 0 and ii < m1 and jj >= 0 and jj < n1:
                        local_result += sm_size[ii % 16, jj % 16] * h[k, l]
            TEMP[i, j, m + 1] = local_result
            next_data[i, j] = TEMP[i, j, m + 1]

        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            TEMP[i, j, m + 1] = TEMP[i, j, 1]
            next_data[i, j] = TEMP[i, j, 1]

        grid.sync()


@cuda.jit
def process_data_partial(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h):
    b, a, d = TEMP.shape
    i, j = cuda.grid(2)
    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2

    # 检查线程索引是否超出范围
    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    # 分配共享内存
    shared_data = cuda.shared.array(shape=(12, 12), dtype=float32)

    for m in range(d - 1):
        if (m % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            curr_temp = data[i, j]
            next_temp = (-vg[i, j, m] * (curr_temp - TEMP[i - 1, j, m]) / dy[i, j] -
                         ug[i, j, m] * (curr_temp - TEMP[i, j - 1, m]) / dx[i, j]) * dt[i, j] + (
                                aa * (TEMP[i + 1, j, m] + TEMP[i - 1, j, m] - 2 * curr_temp) / dy[i, j] ** 2 +
                                bb * (TEMP[i, j + 1, m] + TEMP[i, j - 1, m] - 2 * curr_temp) / dx[i, j] ** 2) * dt[
                            i, j] + curr_temp
            shared_data[cuda.threadIdx.x, cuda.threadIdx.y] = next_temp
        else:
            next_temp = 0
            shared_data[cuda.threadIdx.x, cuda.threadIdx.y] = next_temp

        cuda.syncthreads()  # 同步线程，确保共享内存数据一致性

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            TEMP[i, j, m + 1] = con2(i, j, shared_data, hm, hn, hm2, hn2, h)
            next_data[i, j] = TEMP[i, j, m + 1]




        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            TEMP[i, j, m + 1] = TEMP[i, j, 1]
            next_data[i, j] = TEMP[i, j, 1]

        cuda.syncthreads()  # 再次同步线程，确保数据一致性

#sig2=(float32[:, ::1], float32[:, ::1], float32[:, :, ::1], float32,
 ##float32, float32, float32[:, :, ::1], float32[:, :, ::1], float32,
# float32, float32[:,::1], float32[:, ::1])

@cuda.jit
def process_data_normal2(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h, tnext_data, global_counter, num_blocks):
    # 获取线程块和线程的维度
    block_dim_x, block_dim_y = cuda.blockDim.x, cuda.blockDim.y
    grid_dim_x, grid_dim_y = cuda.gridDim.x, cuda.gridDim.y
    block_x, block_y = cuda.blockIdx.x, cuda.blockIdx.y
    thread_x, thread_y = cuda.threadIdx.x, cuda.threadIdx.y

    # 计算全局索引
    i = block_x * block_dim_x + thread_x
    j = block_y * block_dim_y + thread_y

    # 每个线程块内处理多个数据元素
    stride_x = block_dim_x * grid_dim_x
    stride_y = block_dim_y * grid_dim_y

    # 循环处理多个时间步
    for m in range(TEMP.shape[2] - 1):
        if (m % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        # 在每个线程块内并行处理多个数据元素
        for x in range(i, data.shape[0], stride_x):
            for y in range(j, data.shape[1], stride_y):
                if x > 0 and x < (data.shape[0] - 1) and y > 0 and y < (data.shape[1] - 1):
                    curr_temp = data[x, y]
                    next_temp = (-vg[x, y, m] * (curr_temp - TEMP[x - 1, y, m]) / dy -
                                 ug[x, y, m] * (curr_temp - TEMP[x, y - 1, m]) / dx) * dt + (
                                aa * (TEMP[x + 1, y, m] + TEMP[x - 1, y, m] - 2 * curr_temp) / dy ** 2 +
                                bb * (TEMP[x, y + 1, m] + TEMP[x, y - 1, m] - 2 * curr_temp) / dx ** 2) * dt + curr_temp
                    next_data[x, y] = next_temp
                    tnext_data[x, y] = next_temp
                elif x == data.shape[0] - 1 or y == data.shape[1] - 1:
                    next_temp = 0
                    next_data[x, y] = next_temp
                    tnext_data[x, y] = next_temp

        # 同步线程块内部的所有线程
        cuda.syncthreads()

        # 增加全局计数器
        if thread_x == 0 and thread_y == 0:
            cuda.atomic.add(global_counter, 0, 1)

        # 等待所有线程块完成当前步计算
        while global_counter[0] < num_blocks:
            pass

        # 同步线程块内部的所有线程
        cuda.syncthreads()

        # 在每个线程块内并行计算下一个时间步的局部结果
        for x in range(i, data.shape[0], stride_x):
            for y in range(j, data.shape[1], stride_y):
                if x >= 0 and x < (data.shape[0] - 1) and y >= 0 and y < (data.shape[1] - 1):
                    TEMP[x, y, m + 1] = tnext_data[x, y]
                    next_data[x, y] = TEMP[x, y, m + 1]
                elif x == data.shape[0] - 1 or y == data.shape[1] - 1:
                    TEMP[x, y, m + 1] = TEMP[x, y, 0]
                    next_data[x, y] = TEMP[x, y, 0]

        # 同步线程块内部的所有线程
        cuda.syncthreads()

        # 重置全局计数器
        if thread_x == 0 and thread_y == 0:
            cuda.atomic.add(global_counter, 0, -num_blocks)

        # 同步线程块内部的所有线程
        cuda.syncthreads()




sig5=(float64[:, ::1], float64[:, ::1], float64[:,:,::1],
                     float64, float64, float64, float64[:,:,::1], float64[:,:,::1],
                     float64, float64, float64[:,::1], float64[:,::1], int32)
@cuda.jit(sig5)
def process_data_normal3(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h, tnext_data, m):
    block_dim_x, block_dim_y = cuda.blockDim.x, cuda.blockDim.y
    grid_dim_x, grid_dim_y = cuda.gridDim.x, cuda.gridDim.y
    block_x, block_y = cuda.blockIdx.x, cuda.blockIdx.y
    thread_x, thread_y = cuda.threadIdx.x, cuda.threadIdx.y

#    i = block_x * block_dim_x + thread_x
#    j = block_y * block_dim_y + thread_y


    i, j = cuda.grid(2)

    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    stride_x = block_dim_x * grid_dim_x
    stride_y = block_dim_y * grid_dim_y

    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2

    grid = cuda.cg.this_grid()
    grid.sync()

    for x in range(i, buf_0.shape[0], stride_x):
        for y in range(j, buf_0.shape[1], stride_y):
            if x > 0 and x < (buf_0.shape[0] - 1) and y > 0 and y < (buf_0.shape[1] - 1):
                curr_temp = buf_0[x, y]
                next_temp = (-vg[x, y, m] * (curr_temp - TEMP[x - 1, y, m]) / dy -
                             ug[x, y, m] * (curr_temp - TEMP[x, y - 1, m]) / dx) * dt + (
                            aa * (TEMP[x + 1, y, m] + TEMP[x - 1, y, m] - 2 * curr_temp) / dy ** 2 +
                            bb * (TEMP[x, y + 1, m] + TEMP[x, y - 1, m] - 2 * curr_temp) / dx ** 2) * dt + curr_temp
                buf_1[x, y] = next_temp
                tnext_data[x, y] = next_temp

             ##   if (x == 13) and (y == 15) and (m== 100):
              ##      print("curr_temp")
              ##      print(next_temp)
           ##         print( x )
         ##        print( y )
              ##      print( m )


            elif x == buf_0.shape[0] - 1 or y == buf_0.shape[1] - 1:
                buf_1[x, y] = 0
                tnext_data[x, y] = 0

#    cuda.syncthreads()
    grid.sync()


    for x in range(i, buf_0.shape[0], stride_x):
        for y in range(j, buf_0.shape[1], stride_y):
            if x >= 0 and x < (buf_0.shape[0] - 1) and y >= 0 and y < (buf_0.shape[1] - 1):
                local_result = 0.0
                for k in range(hm):
                    for l in range(hn):
                        ii = x + k - hm2
                        jj = y + l - hn2
                        if ii >= 0 and ii < tnext_data.shape[0] and jj >= 0 and jj < tnext_data.shape[1]:
                            local_result += tnext_data[ii, jj] * h[k, l]
                TEMP[x, y, m + 1] = local_result
#                TEMP[x,y,m+1]= tnext_data[x, y]

                buf_1[x, y] = local_result
            elif x == buf_0.shape[0] - 1 or y == buf_0.shape[1] - 1:
                TEMP[x, y, m + 1] = TEMP[x, y, m]
                buf_1[x, y] = TEMP[x, y, m]

#    cuda.syncthreads()
    grid.sync()








sig2=(float32[:, ::1], float32[:, ::1], float32[:, :, ::1], float32,
 float32, float32, float32[:, :, ::1], float32[:, :, ::1], float32,
 float32, float32[:,::1], float32[:, ::1])
@cuda.jit(sig2)
def process_data_normal2_old(buf_0, buf_1, TEMP, dx, dy, dt, vg, ug, aa, bb, h, tnext_data):
    b, a, d = TEMP.shape
    i, j = cuda.grid(2)



    if (i >= buf_0.shape[0]) or (j >= buf_0.shape[1]):
        return

    grid = cuda.cg.this_grid()

    hm, hn = h.shape
    hm2, hn2 = hm // 2, hn // 2

    grid.sync()
    for m in range(d - 1):
        if (m % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            curr_temp = data[i, j]
            next_temp = (-vg[i, j, m] * (curr_temp - TEMP[i - 1, j, m]) / dy -
                         ug[i, j, m] * (curr_temp - TEMP[i, j - 1, m]) / dx) * dt + (
                         aa * (TEMP[i + 1, j, m] + TEMP[i - 1, j, m] - 2 * curr_temp) / dy ** 2 +
                         bb * (TEMP[i, j + 1, m] + TEMP[i, j - 1, m] - 2 * curr_temp) / dx ** 2) * dt + curr_temp
            next_data[i, j] = next_temp
            tnext_data[i, j]= next_temp
        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            next_temp = 0
            next_data[i, j] = next_temp
            tnext_data[i, j] = next_temp

        grid.sync()


        if (i > 0) & (i < (data.shape[0] - 1)) & (j > 0) & (j < (data.shape[1] - 1)):
            m1, n1 = tnext_data.shape
            local_result = 0.0
            for k in range(hm):
                for l in range(hn):
                    ii = i + k - hm2
                    jj = j + l - hn2
                    if ii >= 0 and ii < m1 and jj >= 0 and jj < n1:
                        local_result += tnext_data[ii, jj] * h[k, l]
            TEMP[i, j, m + 1] = local_result
            next_data[i, j] = TEMP[i, j, m + 1]
        elif (i == data.shape[0] - 1) or (j == data.shape[1] - 1):
            TEMP[i, j, m + 1] = TEMP[i, j, 1]
            next_data[i, j] = TEMP[i, j, 1]

        grid.sync()


# CUDA 核函数用于计算 move_frame 和 move_bg
@cuda.jit
def cal_movef_movebg_kernel2(TEMP, Temp2, x2, y2, CX, Rt, t, X1, X2, move_frame, move_bg, d):
    i, j, m = cuda.grid(3)
    if i < 100 and j < 100 and m < d - 1:
        if m < 40:
            Cx = (-CX[0] * t[m + 1] - 4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * i)
            Cy = (-4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * j)
        elif 40 <= m < 160:
            Cx = (-X1 - CX[99] * (t[m + 1] - t[39]) - 4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * i)
            Cy = (-4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * j)
        else:
            Cx = (-X2 - CX[99] * (t[m + 1] - t[159]) - 4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * i)
            Cy = (-4 * Rt[m + 1] + (8 * Rt[m + 1] / 99) * j)



       ## if (i==20)&(j==20)&(m==150):
         ##   print("Cx:")
         ##   print(Cx)
         ##   print("Cy:")
         ##   print(Cy)
         ##   print("X1:")
         ##   print(X1)
         ##   print("CX[99]:")
         ##   print(CX[99])
         ##   print("Rt[m + 1]")
         ##   print(Rt[m + 1])





        # 执行规则网格插值
        move_frame[j, i, m + 1] = 0
        move_bg[j, i, m + 1] = 0

        dx = x2[1] - x2[0]
        dy = y2[1] - y2[0]

        ix = (Cx - x2[0]) / dx
        iy = (Cy - y2[0]) / dy

        ix_floor = int(math.floor(ix))
        iy_floor = int(math.floor(iy))
        ix_ceil = int(math.ceil(ix))
        iy_ceil = int(math.ceil(iy))

        if ix_floor >= 0 and ix_ceil < x2.size and iy_floor >= 0 and iy_ceil < y2.size:
            move_frame[j, i, m + 1] = ((x2[ix_ceil] - Cx) * (y2[iy_ceil] - Cy) * TEMP[iy_floor, ix_floor, m + 1] +
                                       (Cx - x2[ix_floor]) * (y2[iy_ceil] - Cy) * TEMP[iy_floor, ix_ceil, m + 1] +
                                       (x2[ix_ceil] - Cx) * (Cy - y2[iy_floor]) * TEMP[iy_ceil, ix_floor, m + 1] +
                                       (Cx - x2[ix_floor]) * (Cy - y2[iy_floor]) * TEMP[iy_ceil, ix_ceil, m + 1]) / (dx * dy)

            move_bg[j, i, m + 1] = ((x2[ix_ceil] - Cx) * (y2[iy_ceil] - Cy) * Temp2[iy_floor, ix_floor] +
                                    (Cx - x2[ix_floor]) * (y2[iy_ceil] - Cy) * Temp2[iy_floor, ix_ceil] +
                                    (x2[ix_ceil] - Cx) * (Cy - y2[iy_floor]) * Temp2[iy_ceil, ix_floor] +
                                    (Cx - x2[ix_floor]) * (Cy - y2[iy_floor]) * Temp2[iy_ceil, ix_ceil]) / (dx * dy)

      ##  if (i==44)&(j==55)&(m==150):
        ##    print("move_frame[j, i, m + 1]")
       ##     print(move_frame[j, i, m + 1])
       ##     print(" move_bg[j, i, m + 1]")
       ##     print(move_bg[j, i, m + 1])
        ##    print("y2.size")
    ##        print(y2.size)
      ##      print("x2[ix_ceil]:")
       ##     print(x2[ix_ceil])
      ##      print("x2[ix_floor]:")
   ##     print(x2[ix_floor])
       ##     print("y2[iy_ceil]")
     ##       print(iy_ceil)
     ##       print("iy")
     ##       print(iy)
     ##       print("X1:")
     ##       print(X1)
     ##       print("CX[99]:")
     ##       print(CX[99])
       ##     print("Rt[m + 1]")
      ##      print(Rt[m + 1])
      ##      print("ix_floor")
     ##       print(ix_floor)
     ##       print("TEMP[iy_floor, ix_floor, m + 1]")
     ##       print(TEMP[iy_floor, ix_floor, m + 1])
   ##         print("Temp2[iy_floor, ix_floor]")
    ##        print(Temp2[iy_floor, ix_floor])
    ##        print("A:")
     ##       print(TEMP[iy_floor, ix_ceil, m + 1] )
      ##      print("iy_floor")
     ##       print(iy_floor)
     ##       print("iy_ceil")
     ##       print(iy_ceil)
     ##       print("t:10")
      ##      print(t[10])
           # print("ix")
           # print(ix)
            #print("(x2[ix_ceil] - Cx) * (y2[iy_ceil] - Cy) * TEMP[(iy_floor * (*spank) + ix_floor) + (m + 1) * (*spank) * d]")
            #print((x2[ix_ceil] - Cx) * (y2[iy_ceil] - Cy) * TEMP[iy_floor, ix_floor, m + 1])
            #print("(Cx - x2[ix_floor]) * (y2[iy_ceil] - Cy) * TEMP[(iy_ceil * (*spank) + ix_floor) + (m + 1) * (*spank) * d]")
           # print(( x2[ix_floor])    )
      ##     print(move_bg[j, i, m + 1] )
    ##        print("OK")







# 使用 CUDA 计算 move_frame 和 move_bg 的函数
def cal_movef_movebg(TEMP, CX, Rt, t, x2, y2, Temp2, X1, X2,stream):
    d = 200
    move_frame = np.zeros((100, 100, d))
    move_bg = np.zeros((100, 100, d))

    # 打印数据以进行调试
#    print("TEMP:", TEMP)
#    print("CX:", CX)
 #   print("Rt:", Rt)
 #   print("t:", t)
 #   print("x2:", x2)
 #   print("y2:", y2)
#    print("Temp2:", Temp2)
 #   print("X1:", X1)
  #  print("X2:", X2)

    # 分配设备内存
    TEMP_dev = cuda.to_device(TEMP,stream=stream)
    Temp2_dev = cuda.to_device(Temp2,stream=stream)
    x2_dev = cuda.to_device(x2,stream=stream)
    y2_dev = cuda.to_device(y2,stream=stream)
    CX_dev = cuda.to_device(CX,stream=stream)
    Rt_dev = cuda.to_device(Rt,stream=stream)
    t_dev = cuda.to_device(t,stream=stream)
    move_frame_dev = cuda.to_device(move_frame,stream=stream)
    move_bg_dev = cuda.to_device(move_bg,stream=stream)

    # 定义块和网格大小
    threads_per_block = (16, 16, 4)
    blocks_per_grid = (math.ceil(100 / threads_per_block[0]),
                       math.ceil(100 / threads_per_block[1]),
                       math.ceil(d / threads_per_block[2]))

    # 启动核函数
    cal_movef_movebg_kernel2[blocks_per_grid, threads_per_block,stream](TEMP_dev, Temp2_dev, x2_dev, y2_dev, CX_dev, Rt_dev, t_dev, X1, X2, move_frame_dev, move_bg_dev, d)

    # 将结果复制回主机
    move_frame = move_frame_dev.copy_to_host(stream=stream)
    move_bg = move_bg_dev.copy_to_host(stream=stream)

    # 打印结果以进行调试
  #  print("move_frame:", move_frame)
 #   print("move_bg:", move_bg)

    return move_frame, move_bg



def cal_movef_movebg_old(TEMP, CX, Rt, t, x2, y2, Temp2, X1, X2):
    d = 200
    move_frame = np.zeros((100, 100, d))
    move_bg = np.zeros((100, 100, d))
    for m in range(d - 1):
        if m < 40:
            Cx = np.linspace(-CX[0] * t[m + 1] - 4 * Rt[m + 1], -CX[0] * t[m + 1] + 4 * Rt[m + 1], 100)
            Cy = np.linspace(-4 * Rt[m + 1], 4 * Rt[m + 1], 100)
            move_frame[:, :, m + 1] = regular_grid_interpolator(x2, y2, TEMP[:, :, m + 1], Cx, Cy)
            move_bg[:, :, m + 1] = regular_grid_interpolator(x2, y2, Temp2, Cx, Cy)
        elif 40 <= m < 160:
            Cx = np.linspace(-X1 - CX[99] * (t[m + 1] - t[39]) - 4 * Rt[m + 1],
                             -X1 - CX[99] * (t[m + 1] - t[39]) + 4 * Rt[m + 1], 100)
            Cy = np.linspace(-4 * Rt[m + 1], 4 * Rt[m + 1], 100)
            move_frame[:, :, m + 1] = regular_grid_interpolator(x2, y2, TEMP[:, :, m + 1], Cx, Cy)
            move_bg[:, :, m + 1] = regular_grid_interpolator(x2, y2, Temp2, Cx, Cy)
        elif m >= 160:
            Cx = np.linspace(-X2 - CX[99] * (t[m + 1] - t[159]) - 4 * Rt[m + 1],
                             -X2 - CX[99] * (t[m + 1] - t[159]) + 4 * Rt[m + 1], 100)
            Cy = np.linspace(-4 * Rt[m + 1], 4 * Rt[m + 1], 100)
            move_frame[:, :, m + 1] = regular_grid_interpolator(x2, y2, TEMP[:, :, m + 1], Cx, Cy)
            move_bg[:, :, m + 1] = regular_grid_interpolator(x2, y2, Temp2, Cx, Cy)

    return move_frame, move_bg




def SST_backup_stcc_25_UV(rx, ry, sc, mv, U, V, dtdy, Rend, A, T, lat, rho0, T0):

    CX, ug, vg = uv_cal_UV(rx, ry, sc, mv, U, V, Rend, A, T, lat, rho0)

    a = rx
    b = ry
    d = 200
    pt = T * 24 * 60 * 60
    xx = np.linspace(-sc, sc, a) * Rend
    dx = xx[1] - xx[0]
    yy = np.linspace(-sc, sc, b) * Rend
    dy = yy[1] - yy[0]
    t = np.linspace(0, pt, d)
    dt = t[1] - t[0]
#    x2, y2 = meshgrid_numba(xx, yy)
    x2,y2=xx,yy

    if dtdy < 0:
        Tmax = T0 - dtdy * sc * Rend
        Tmin = T0 + dtdy * sc * Rend
    else:
        Tmax = T0 + dtdy * sc * Rend
        Tmin = T0 - dtdy * sc * Rend

    Temp = np.linspace(Tmax, Tmin, b).reshape(-1, 1)
    Temp2 = np.repeat(Temp, a, axis=1)

    Rt = (1 / 40 * np.arange(1, 201) * (np.arange(1, 201) <= 40) +
          1 * ((np.arange(1, 201) > 40) & (np.arange(1, 201) <= 160)) +
          (-1 / 40) * (np.arange(1, 201) - 200) * (np.arange(1, 201) > 160)) * Rend

    TEMP = np.zeros((a, b, d))
    TEMP[:, :, 0] = Temp2

    X1 = CX[0] * t[40] + 0.5 * ((CX[1] - CX[0]) / (t[1] - t[0])) * t[40] ** 2
    X2 = X1 + CX[159] * (t[159] - t[40])

    move_frame = np.zeros((100, 100, 200))
    move_bg = np.zeros((100, 100, 200))

    h = 1 / 10.8 * np.array([[1.4, 1, 1], [2, 1, 1], [1.4, 1, 1]])
    aa = 2000
    bb = 2000




    buf_0 = cuda.to_device(Temp2)
    buf_1 = cuda.device_array_like(buf_0)
    buf_2 = cuda.to_device(TEMP)
    buf_3 = cuda.to_device(h)



    process_data[(32,32), (16,16)](buf_0, buf_1, buf_2, dx, dy, dt, vg, ug, aa, bb, buf_3)


    TEMP = buf_2.copy_to_host()


    move_frame, move_bg = cal_movef_movebg(TEMP, CX, Rt, t, x2, y2, Temp2, X1, X2)

    move_an = move_frame - move_bg

    return move_frame, move_bg, move_an





def Var_ex(c, V, dtdy, R, SLA, T, lat, rho, t, U):
    j = 50
#    span = np.arange(10, 21, 2)

#    T_r=np.zeros((100,100,len(span)))

#    for i in range(len(span)):
    i=250
    _, _, move_an = SST_backup_stcc_25_UV(i, i, j, c, U, 0, dtdy, R * 1000, SLA, T, lat, rho, t)
    N = 199
    result = np.random.randint(N - 1, size=(20000, 1))
    t_r = np.mean(move_an[:, :, result], axis=2)
    T_r=t_r[:,:,0]



#    print(str(i))
#    plt.contourf(move_an[:, :, 10])
#    plt.show()
#    plt.gcf().set_size_inches(4, 4)  # 设置图像大小为 4x4 inches

#    sys.exit()
    # 保存为400x400的jpg图片
#    strr = 'C:\\MAIN_pg\\pytorch\\test\\' + str(i) + '.jpg'
#    plt.savefig(strr, dpi=100)  # dpi参数控制图像的分辨率

    return T_r



@cuda.jit(device=True)
def linear_interpolation(x, y, z, xi, yi):
    x0, x1 = int(xi), int(xi) + 1
    y0, y1 = int(yi), int(yi) + 1

    x0 = min(max(x0, 0), x.shape[0] - 1)
    x1 = min(max(x1, 0), x.shape[0] - 1)
    y0 = min(max(y0, 0), y.shape[1] - 1)
    y1 = min(max(y1, 0), y.shape[1] - 1)

    Q11 = z[x0, y0]
    Q21 = z[x1, y0]
    Q12 = z[x0, y1]
    Q22 = z[x1, y1]

    return (Q11 * (x1 - xi) * (y1 - yi) +
            Q21 * (xi - x0) * (y1 - yi) +
            Q12 * (x1 - xi) * (yi - y0) +
            Q22 * (xi - x0) * (yi - y0))


# CUDA kernel function for grid interpolation
@cuda.jit
def interpolate_kernel(move_frame, move_bg, TEMP, x2, y2, Cx, Cy):
    # Calculate global thread position
    idx, idy = cuda.grid(2)

    if idx < move_frame.shape[0] and idy < move_frame.shape[1]:
        # Perform interpolation for each slice
        for m in range(move_frame.shape[2]):
            move_frame[idx, idy, m] = linear_interpolation(x2, y2, TEMP[:, :, m + 1], Cx, Cy)
            move_bg[idx, idy, m] = linear_interpolation(x2, y2, TEMP[:, :, 0], Cx, Cy)




def cal_movef_movebg_parallel(TEMP, CX, Rt, t, x2, y2, Temp2, X1, X2):
    d = 200
    move_frame = np.zeros((100, 100, d), dtype=np.float64)
    move_bg = np.zeros((100, 100, d), dtype=np.float64)

    # Allocate memory on device
    move_frame_global_mem = cuda.to_device(move_frame)
    move_bg_global_mem = cuda.to_device(move_bg)
    TEMP_global_mem = cuda.to_device(TEMP)
    x2_global_mem = cuda.to_device(x2)
    y2_global_mem = cuda.to_device(y2)
    Cx_global_mem = cuda.to_device(CX)
    Rt_global_mem = cuda.to_device(Rt)
    t_global_mem = cuda.to_device(t)
    Temp2_global_mem = cuda.to_device(Temp2)

    # Define block and grid dimensions
    threads_per_block = (16, 16, 4)
    blocks_per_grid = (math.ceil(100 / threads_per_block[0]),
                       math.ceil(100 / threads_per_block[1]),
                       math.ceil(d / threads_per_block[2]))

    # Launch CUDA kernel
    interpolate_kernel[blocks_per_grid, threads_per_block](move_frame_global_mem, move_bg_global_mem,
                                                           TEMP_global_mem, x2_global_mem, y2_global_mem,
                                                           Cx_global_mem, Rt_global_mem, t_global_mem, X1, X2, d)

    # Copy results back to host
    move_frame = move_frame_global_mem.copy_to_host()
    move_bg = move_bg_global_mem.copy_to_host()

    # Free device memory
    move_frame_global_mem.free()
    move_bg_global_mem.free()
    TEMP_global_mem.free()
    x2_global_mem.free()
    y2_global_mem.free()
    Cx_global_mem.free()
    Rt_global_mem.free()
    t_global_mem.free()
    Temp2_global_mem.free()

    return move_frame, move_bg




