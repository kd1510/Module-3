from numba import njit, prange, cuda

import minitorch
import minitorch.fast_ops

from time import perf_counter

from minitorch.tensor_ops import SimpleBackend


# Basic sum cuda test

@cuda.jit()
def kernel(data):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # print("thread global id:",  x, "data value: ", data[x])

    # if x < len(data):
    data[x] *= 2
    # print(x, y)

# local = cuda.local.array(5)

# threadsperblock = (4, 3)
# blockerpergrid = (1, 3)
# kernel[blockspergrid, threadsperblock](in, local[1])
# kernel()
#

import numpy as np

data = np.arange(256*10000)
threadsperblock = 64

blockspergrid = (data.size + (threadsperblock-1)) // threadsperblock

print('data size: ', data.size)
print("blocks per grid: ", blockspergrid)

kernel[blockspergrid, threadsperblock](data)

print(data)



data2 = np.arange(256*10000)
for i in range(len(data2)):
    data2[i] *= 2

print(data2)

# def test1():
#     z = 0
#     for i in prange(10000000):
#         z += i / 100

#     return z

# fn_opt = njit()(test1)

# start = perf_counter()
# test1()
# elapsed = perf_counter() - start
# print(f"Non JIT: {elapsed:.4f} seconds")


# start = perf_counter()
# fn_opt()
# elapsed = perf_counter() - start
# print(f"JIT: {elapsed:.4f} seconds")

# MAP
# print("MAP")
# tmap = minitorch.fast_ops.tensor_map(njit()(lambda x: x + 1))
# bmap = minitorch.tensor_map(lambda x: x + 1)

# out, a = minitorch.zeros((10000,)), minitorch.zeros((10000,))
# tmap(*out.tuple(), *a.tuple())
# print(tmap.parallel_diagnostics(level=3))

# start = perf_counter()

# for _ in range(1000):
#     bmap(*out.tuple(), *a.tuple())

# elapsed = perf_counter() - start
# print(f"Non JIT: {elapsed:.4f} seconds")


# start = perf_counter()
# for _ in range(10000):
#     tmap(*out.tuple(), *a.tuple())

# elapsed = perf_counter() - start
# print(f"JIT: {elapsed:.4f} seconds")


# ZIP
# print("ZIP")
# out, a, b = minitorch.zeros((1000,)), minitorch.zeros((1000,)), minitorch.zeros((1000,))
# tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))
# print(tzip.parallel_diagnostics(level=3))

# start = perf_counter()

# for _ in range(1000):
#     tzip(*out.tuple(), *a.tuple(), *b.tuple())

# elapsed = perf_counter() - start
# print(f"JIT: {elapsed:.4f} seconds")

# start = perf_counter()
# for _ in range(1000):
#     minitorch.tensor_zip(minitorch.operators.eq)(*out.tuple(), *a.tuple(), *b.tuple())

# elapsed = perf_counter() - start
# print(f"Non JIT: {elapsed:.4f} seconds")

# # REDUCE
# print("REDUCE")
# out, a = minitorch.zeros((100,)), minitorch.zeros((20000,))
# treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))

# treduce(*out.tuple(), *a.tuple(), 0)
# print(treduce.parallel_diagnostics(level=3))

# start = perf_counter()

# for _ in range(1000):
#     treduce(*out.tuple(), *a.tuple(), 0)

# elapsed = perf_counter() - start
# print(f"JIT: {elapsed:.4f} seconds")

# start = perf_counter()
# for _ in range(1000):
#     minitorch.tensor_reduce(minitorch.operators.add)(*out.tuple(), *a.tuple(), 0)

# elapsed = perf_counter() - start
# print(f"Non JIT: {elapsed:.4f} seconds")


# # MM
# print("MATRIX MULTIPLY")

# import torch

# data = [1, 2, 3, 4, 5, 6]
# x = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(3, 2)
# y = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)

# torch_result = x @ y
# print(torch_result)

# out, a, b = (
#     minitorch.zeros((1, 3, 3)),
#     minitorch.Tensor.make(data, (1, 3, 2), backend=SimpleBackend),
#     minitorch.Tensor.make(data, (1, 2, 3), backend=SimpleBackend),
# )
# tmm = minitorch.fast_ops.tensor_matrix_multiply

# tmm(*out.tuple(), *a.tuple(), *b.tuple())
# # print(tmm.parallel_diagnostics(level=3))

# print(out)
