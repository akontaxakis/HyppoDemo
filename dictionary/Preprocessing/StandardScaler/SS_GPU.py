import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn.base import BaseEstimator, TransformerMixin

mod = SourceModule("""
__global__ void standardize(float *data, float *mean, float *std, int rows, int cols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows)
    {
        data[idx] = (data[idx] - mean[col]) / std[col];
    }
}
""")

standardize = mod.get_function("standardize")


class GPU__StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0).astype(np.float32)
        self.scale_ = np.std(X, axis=0).astype(np.float32)
        return self

    def transform(self, X):
        rows, cols = X.shape
        #print(type(X))
        X_gpu = gpuarray.to_gpu(np.array(X, dtype=np.float32))
        mean_gpu = gpuarray.to_gpu(self.mean_)
        scale_gpu = gpuarray.to_gpu(self.scale_)

        standardize(X_gpu, mean_gpu, scale_gpu, np.int32(rows), np.int32(cols),
                    block=(1024, 1, 1), grid=((rows * cols + 1023) // 1024, 1))

        return X_gpu.get()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
