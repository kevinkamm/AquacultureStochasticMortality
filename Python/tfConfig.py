import tensorflow as tf

sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(cuda_version)
cudnn_version = sys_details["cudnn_version"]  
print(cudnn_version)
cpu_compiler = sys_details["cpu_compiler"]  
print(cpu_compiler)
cuda_compute_capabilities = sys_details["cuda_compute_capabilities"]  
print(cuda_compute_capabilities)

import torch
print(torch.version.cuda)

from numba import cuda
cuda.cudadrv.libs.test()