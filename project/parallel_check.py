from numba import njit, prange

import minitorch
import minitorch.fast_ops

from time import perf_counter

def test1():
    z = 0
    for i in prange(10000000):
        z += i / 100

    return z

fn_opt = njit()(test1)

start = perf_counter()
test1()
elapsed = perf_counter() - start
print(f"Non JIT: {elapsed:.4f} seconds")


start = perf_counter()
fn_opt()
elapsed = perf_counter() - start
print(f"JIT: {elapsed:.4f} seconds")

# # MAP
# print("MAP")
# tmap = minitorch.fast_ops.tensor_map(njit()(minitorch.operators.id))
# out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
# tmap(*out.tuple(), *a.tuple())
# print(tmap.parallel_diagnostics(level=3))

# # ZIP
# print("ZIP")
# out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
# tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))

# tzip(*out.tuple(), *a.tuple(), *b.tuple())
# print(tzip.parallel_diagnostics(level=3))

# # REDUCE
# print("REDUCE")
# out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
# treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))

# treduce(*out.tuple(), *a.tuple(), 0)
# print(treduce.parallel_diagnostics(level=3))


# # MM
# print("MATRIX MULTIPLY")
# out, a, b = (
#     minitorch.zeros((1, 10, 10)),
#     minitorch.zeros((1, 10, 20)),
#     minitorch.zeros((1, 20, 10)),
# )
# tmm = minitorch.fast_ops.tensor_matrix_multiply

# tmm(*out.tuple(), *a.tuple(), *b.tuple())
# print(tmm.parallel_diagnostics(level=3))
