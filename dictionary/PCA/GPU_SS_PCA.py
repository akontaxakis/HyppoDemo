import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn.base import BaseEstimator, TransformerMixin

mod = SourceModule("""
__global__ void preprocess(float *data, float *mean, float *std, int rows, int cols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows)
    {
        data[idx] = (data[idx] - mean[col]) / std[col];
    }
}
""", options=["-Wno-deprecated-gpu-targets", "-Xcompiler", "-Wall"])

preprocess = mod.get_function("preprocess")

class GPU_StandardScaler__PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.scale_ = None
        self.components_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=None).astype(np.float32)
        self.scale_ = np.std(X, axis=None).astype(np.float32)

        rows, cols = X.shape
        X_gpu = gpuarray.to_gpu(np.array(X, dtype=np.float32))
        mean_gpu = gpuarray.to_gpu(self.mean_)
        scale_gpu = gpuarray.to_gpu(self.scale_)

        preprocess(X_gpu, mean_gpu, scale_gpu, np.int32(rows), np.int32(cols),
                   block=(1024, 1, 1), grid=((rows * cols + 1023) // 1024, 1))

        X_centered = X_gpu.get()
        cov = np.cov(X_centered, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov)
        sorted_indices = np.argsort(eigvals)[::-1][:self.n_components]
        self.components_ = eigvecs[:, sorted_indices].T
        return self

    def transform(self, X):
        X_standardized = (X - self.mean_) / self.scale_
        return X_standardized.dot(self.components_.T)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
