import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn.base import BaseEstimator, TransformerMixin

mod = SourceModule("""
__global__ void subtract_mean(float *a, float *mean, int rows, int cols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows)
    {
        a[idx] -= mean[col];
    }
}

__global__ void compute_cov(float *a, float *cov, int rows, int cols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int col1 = idx / cols;
    int col2 = idx % cols;

    if (col1 < cols && col2 < cols)
    {
        float sum = 0.0;

        for (int i = 0; i < rows; i++)
        {
            sum += a[i * cols + col1] * a[i * cols + col2];
        }

        cov[idx] = sum / (rows - 1);
    }
}
""")

subtract_mean = mod.get_function("subtract_mean")
compute_cov = mod.get_function("compute_cov")


class GPU__PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: object) -> object:
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.nan_to_num(X)
        rows, cols = X.shape
        X_gpu = gpuarray.to_gpu(np.array(X, dtype=np.float32))
        mean_gpu = gpuarray.to_gpu(np.mean(X, axis=0).astype(np.float32))

        subtract_mean(X_gpu, mean_gpu, np.int32(rows), np.int32(cols),
                      block=(1024, 1, 1), grid=((rows * cols + 1023) // 1024, 1))

        cov_gpu = gpuarray.empty((cols, cols), dtype=np.float32)
        compute_cov(X_gpu, cov_gpu, np.int32(rows), np.int32(cols),
                    block=(1024, 1, 1), grid=((cols * cols + 1023) // 1024, 1))

        eigvals, eigvecs = np.linalg.eigh(cov_gpu.get())
        sorted_indices = np.argsort(eigvals)[::-1][:self.n_components]
        self.components_ = eigvecs[:, sorted_indices].T
        return self

    def transform(self, X):
        X = np.nan_to_num(X)
        return X.dot(self.components_.T)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
