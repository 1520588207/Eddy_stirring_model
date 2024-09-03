import sys
import numpy as np
import scipy.io
from scipy.io import loadmat
import math
import gc
#import nvtx
from numba import cuda, float64, int32
import matplotlib.pyplot as plt
import time
from utils_gpu_lg import uv_cal_UV, process_data_normal2, meshgrid_numba, calculate_velocity2, cal_movef_movebg, sw_f, con2, calculate_velocity, process_data_normal3
from utils_gpu_lg import cal_movef_movebg_parallel
from numba import njit

@njit(parallel=True)
def compute_mean(move_an, result):
    X, Y, _ = move_an.shape
    num_results = len(result)
    t_r = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            total = 0.0
            # Compute the sum of the selected elements
            for idx in result:
                total += move_an[x, y, idx]
            # Calculate the mean
            t_r[x, y] = total / num_results

    return t_r

@cuda.jit
def compute_mean_kernel(move_an, result, t_r):
    x, y = cuda.grid(2)
    if x < t_r.shape[0] and y < t_r.shape[1]:
        total = 0.0
        num_results = result.size
        for idx in range(num_results):
            total += move_an[x, y, result[idx]]
        t_r[x, y] = total / num_results




# Function to process each cell in a stream
def process_cell(i, j, k, stream):
    if (~np.isnan(grad_y_gl[i, j])) & (~np.isnan(Rmap_gl[i, j])):

        #with nvtx.annotate("init_p", color="yellow"):

         if grad_y_gl[i, j] < 0:
            Tmax = t_gl[i, j] - grad_y_gl[i, j] * sc * Rmap_gl[i, j]
            Tmin = t_gl[i, j] + grad_y_gl[i, j] * sc * Rmap_gl[i, j]
            Temp = np.linspace(Tmax, Tmin, span[k], dtype=np.float64).reshape(-1, 1)
            Temp2 = np.repeat(Temp, span[k], axis=1)
         else:
            Tmax = t_gl[i, j] + grad_y_gl[i, j] * sc * Rmap_gl[i, j]
            Tmin = t_gl[i, j] - grad_y_gl[i, j] * sc * Rmap_gl[i, j]
            Temp = np.linspace(Tmin, Tmax, span[k], dtype=np.float64).reshape(-1, 1)
            Temp2 = np.repeat(Temp, span[k], axis=1)



         TEMP = np.zeros((span[k], span[k], 200), dtype=np.float64)
         TEMP[:, :, 0] = Temp2.astype(np.float64)

         T = 100
         t = np.linspace(0, T, 200) * 24 * 60 * 60
         dt = t[1] - t[0]
         Rt = (1 / 40 * np.arange(201) * (np.arange(201) <= 40) + 1 * ((np.arange(201) > 40) & (np.arange(201) <= 160)) +
              (-1 / 40) * (np.arange(201) - 200) * (np.arange(201) > 160)) * Rmap_gl[i, j]
         At = (1 / 40 * np.arange(201) * (np.arange(201) <= 40) + 1 * ((np.arange(201) > 40) & (np.arange(201) <= 160)) +
              (-1 / 40) * (np.arange(201) - 200) * (np.arange(201) > 160)) * A_gl[i, j]
         xx2 = np.linspace(-sc, sc, span[k]) * Rmap_gl[i, j]
         dx = xx2[1] - xx2[0]
         yy2 = np.linspace(-sc, sc, span[k]) * Rmap_gl[i, j]
         dy = yy2[1] - yy2[0]

         [x2, y2] = meshgrid_numba(xx2, yy2)

         rho0s = rho_gl[i, j]
         CX = Dlon_gl[i, j] * np.ones(201)
         X1 = CX[0] * t[39]
         X2 = X1 + CX[159] * (t[159] - t[39])

         f = sw_f(yy[i, j])

         ug = np.zeros((span[k], span[k], 200), dtype=np.float64)
         vg = np.zeros((span[k], span[k], 200), dtype=np.float64)
         ugg = np.zeros((span[k] - 1, span[k] - 1, 200), dtype=np.float64)
         vgg = np.zeros((span[k] - 1, span[k] - 1, 200), dtype=np.float64)

         buf_0 = cuda.to_device(ugg, stream=stream)
         buf_1 = cuda.to_device(vgg, stream=stream)

        # Adjust threads and blocks for CUDA limits
         threads_per_block = (16, 16)

         overload = calculate_velocity.overloads[(float64[:, :, ::1], float64[:, :, ::1], float64[::1], float64[:, ::1],
                                                  float64[:, ::1], float64[::1], float64[::1], float64[::1], float64,
                                                  float64, float64, float64, float64, float64, float64, float64, float64)]

         max_blocks = overload.max_cooperative_grid_blocks(blockdim=threads_per_block)
         print("Maximum cooperative grid blocks:", max_blocks)

         sqrt_floor = math.floor(math.sqrt(max_blocks))

         blocks_per_grid_x = min(int(np.ceil(buf_0.shape[0] / threads_per_block[0])), sqrt_floor)
         blocks_per_grid_y = min(int(np.ceil(buf_0.shape[1] / threads_per_block[1])), sqrt_floor)
         blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

         calculate_velocity[blocks_per_grid, threads_per_block, stream](buf_0, buf_1, float64(At), float64(x2), float64(y2), float64(t),
                                                                        float64(CX), float64(Rt), float64(g), float64(f), float64(rho0s),
                                                                        float64(U_gl[i, j]), float64(0), float64(dx), float64(dy), float64(X1), float64(X2))

         ug1_o = buf_0.copy_to_host(stream=stream)
         vg1_o = buf_1.copy_to_host(stream=stream)

        #with nvtx.annotate("append", color="blue"):

         ug[:, :, :] = np.pad(ug1_o, ((0, 1), (0, 1), (0, 0)), mode='edge')
         vg[:, :, :] = np.pad(vg1_o, ((0, 1), (0, 1), (0, 0)), mode='edge')


        # Buffer arrays for temperature data
         buf_0 = cuda.to_device(Temp2, stream=stream)
         buf_1 = cuda.device_array_like(buf_0, stream=stream)

         buf_2 = cuda.to_device(TEMP, stream=stream)
         vg_dev = cuda.to_device(vg, stream=stream)
         ug_dev = cuda.to_device(ug, stream=stream)

         tdata = np.zeros(buf_0.shape, dtype=np.float64)
         tdata_dev = cuda.to_device(tdata, stream=stream)

         overload = process_data_normal3.overloads[(float64[:, ::1], float64[:, ::1], float64[:, :, ::1],
                                                   float64, float64, float64, float64[:, :, ::1], float64[:, :, ::1],
                                                   float64, float64, float64[:, ::1], float64[:, ::1], int32)]

         max_blocks = overload.max_cooperative_grid_blocks(blockdim=threads_per_block)
         sqrt_floor = math.floor(math.sqrt(max_blocks))

         blocks_per_grid_x = min(int(np.ceil(buf_0.shape[0] / threads_per_block[0])), sqrt_floor)
         blocks_per_grid_y = min(int(np.ceil(buf_0.shape[1] / threads_per_block[1])), sqrt_floor)
         blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

         nt = 200
         for m in range(nt - 1):
            if m % 2 == 0:
                process_data_normal3[blocks_per_grid, threads_per_block, stream](buf_0, buf_1, buf_2, dx, dy, dt, vg_dev, ug_dev, aa, bb, h, tdata_dev, int32(m))
            else:
                process_data_normal3[blocks_per_grid, threads_per_block, stream](buf_1, buf_0, buf_2, dx, dy, dt, vg_dev, ug_dev, aa, bb, h, tdata_dev, int32(m))

         T = buf_2.copy_to_host(stream=stream)


         #with nvtx.annotate("movebg",color="black"):
         move_frame, move_bg = cal_movef_movebg(T, CX, Rt, t, xx2, yy2, Temp2, X1, X2,stream)



         #with nvtx.annotate("t_r", color="yellow"):
         N = 199
         result = np.random.randint(N - 1, size=(20000,), dtype=np.int32)
         move_an = move_frame - move_bg
         move_an_gpu = cuda.to_device(move_an, stream=stream)
         result_gpu = cuda.to_device(result, stream=stream)

         # 初始化结果数组
         t_r_gpu = cuda.device_array((move_an.shape[0], move_an.shape[1]), dtype=np.float32)

         # CUDA grid dimensions
         threadsperblock = (32, 32)
         blockspergrid_x = int(np.ceil(move_an.shape[0] / threadsperblock[0]))
         blockspergrid_y = int(np.ceil(move_an.shape[1] / threadsperblock[1]))
         blockspergrid = (blockspergrid_x, blockspergrid_y)

         # 执行 CUDA kernel
         compute_mean_kernel[blockspergrid, threadsperblock, stream](move_an_gpu, result_gpu, t_r_gpu)

         # 将结果从 GPU 复制回 CPU
         t_r = t_r_gpu.copy_to_host(stream=stream)


        #with nvtx.annotate("save", color="blue"):
         data_to_save = {
            't_r':t_r,
         }

         scipy.io.savemat(f"{k}_{i}_{j}.mat", data_to_save)



start_time = time.time()

#with nvtx.annotate("init", color="blue"):
# Load data
data = loadmat('data_co.mat')
grad_y_gl = data['grad_y_co']
t_gl = data['t_co']
Rmap_gl = data['Rmap_co']
A_gl = data['Amap_co']
Dlon_gl = data['Dlon_co']
U_gl = data['Umean_co']
rho_gl = data['rho_co']
yy = data['yyp']

# Constants
g = np.float64(9.8)
sc = 50
span = np.arange(2500, 3000, 10)
h = 1 / 10.8 * np.array([[1.4, 1, 1], [2, 1, 1], [1.4, 1, 1]], dtype=np.float64)
n, m = h.shape
aa = np.float64(2000)
bb = np.float64(2000)

# Main processing loop with streams
streams = []
num_streams = 8  # Number of streams to use

for i in range(num_streams):
    streams.append(cuda.stream())


for i in range(30,42):
     for j in range(20,21):
         for k in range(0, 1):
            stream_index =0
            process_cell(i, j, k, streams[stream_index])

          #  with nvtx.annotate("ts", color="blue"):
           #     time.sleep(0.001)

elapsed_time = time.time() - start_time
print(f"{elapsed_time:.2f} seconds.")

# Clean up streams
for stream in streams:
    stream.synchronize()
