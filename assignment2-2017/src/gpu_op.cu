#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */

__global__ void array_set_kernel(int num, float* data, float value){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= num) {
    return;
  }
  data[y] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < arr->ndim; i++) {
    n *= arr->shape[i];
  }
  float *array_data = (float *)arr->data;
  dim3 threads;
  dim3 blocks;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  array_set_kernel<<<blocks, threads>>>(n, array_data, value);
  return 0;
}

__global__ void broadcast_to_kernel(int in_num, int out_num, const float* input, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= in_num){
    return;
  }
  for(int i=y; i< out_num; i += in_num) {
    output[i] = input[y];
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int in_num = 1;
  for (int i = 0; i < input->ndim; i++) {
    in_num *= input->shape[i];
  }
  int out_num = 1;
  for (int i = 0; i < output->ndim; i++) {
    out_num *= output->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  if (in_num <= 1024) {
    threads.x = in_num;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (in_num + 1023) / 1024;
  }
  broadcast_to_kernel<<<blocks, threads>>>(in_num, out_num, input_data, output_data);
  return 0;
}

__global__ void reduce_sum_axis_zero_kernel(int in_num, int out_num, const float *input_data, float *output_data) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= out_num) {
    return;
  }
  output_data[y] = 0;
  for (int i = y; i < in_num; i += out_num) {
    output_data[y] += input_data[i];
  }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int in_num = 1;
  for (int i = 0; i < input->ndim; i++) {
    in_num *= input->shape[i];
  }
  int reduced_num = 1;
  for (int i = 1; i < input->ndim; i++) {
    reduced_num *= input->shape[i];
  }
  int out_num = 1;
  for (int i = 0; i < output->ndim; i++) {
    out_num *= output->shape[i];
  }
  assert(reduced_num == out_num);
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  if (out_num <= 1024) {
    threads.x = out_num;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (out_num + 1023) / 1024;
  }
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(in_num, out_num, input_data, output_data);
  return 0;
}

__global__ void matrix_elementwise_add_kernel(int n, const float* input_a, const float* input_b, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input_a[y] + input_b[y];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < matA->ndim; i++) {
    n *= matA->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data_a = (const float *)matA->data;
  const float *input_data_b = (const float *)matB->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  matrix_elementwise_add_kernel<<<blocks, threads>>>(n, input_data_a, input_data_b, output_data);
  return 0;
}

__global__ void matrix_elementwise_add_by_const_kernel(int n, const float* input, float val, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input[y] + val;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < input->ndim; i++) {
    n *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  matrix_elementwise_add_by_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
  return 0;
}

__global__ void matrix_elementwise_multiply_kernel(int n, const float* input_a, const float* input_b, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input_a[y] * input_b[y];
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < matA->ndim; i++) {
    n *= matA->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data_a = (const float *)matA->data;
  const float *input_data_b = (const float *)matB->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  matrix_elementwise_multiply_kernel<<<blocks, threads>>>(n, input_data_a, input_data_b, output_data);
  return 0;
}

__global__ void matrix_elementwise_multiply_by_const_kernel(int n, const float* input, float val, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input[y] * val;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < input->ndim; i++) {
    n *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  matrix_elementwise_multiply_by_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
  return 0;
}

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transA, bool transB){
  float Cvalue = 0.0;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;
  if(!transA && !transB){
    assert(colA == rowB);
    if(r >= rowA || c >= colB) return;
    for (int e = 0; e < colA; ++e) {
      Cvalue += (A[r * colA + e]) * (B[e * colB + c]);
    }
    C[r * colB + c] = Cvalue;
  } else if(transA && !transB){
    assert(rowA == rowB);
    if(r>=colA || c >= colB) return;
    for(int e=0; e<rowA; ++e){
      Cvalue += (A[e * rowA + r]) * (B[e * colB + c]);
    }
    C[r * colB + c] = Cvalue;
  } else if(!transA && transB){
    assert(colA == colB);
    if(r >= rowA || c >= rowB) return;
    for (int e = 0; e < colA; ++e){
      Cvalue += (A[r * colA + e]) * (B[c * colB + e]);
    } 
    C[r * rowB + c] = Cvalue;
  } else if(transA && transB){
    assert(rowA == colB);
    if(r >= colA || c >= rowB) return;
    for (int e = 0; e < rowA; ++e){
      Cvalue += (A[e * colA + r]) * (B[c * colB + e]);
    }
    C[r * rowB + c] = Cvalue;
  }
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  int rowA = matA->shape[0];
  int colA = matA->shape[1];
  int rowB = matB->shape[0];
  int colB = matB->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;

  const int BLOCK_SIZE = 16;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((max(rowA, colA) + dimBlock.x - 1) / dimBlock.x, (max(rowB, colB) + dimBlock.y - 1) / dimBlock.y);
  matrix_multiply_kernel<<<dimGrid, dimBlock>>>(matA_data, matB_data, matC_data, rowA, colA, rowB, colB, transposeA, transposeB);
  return 0;
}

__global__ void relu_kernel(int n, const float* input, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input[y] > 0 ? input[y] : 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < input->ndim; i++) {
    n *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  relu_kernel<<<blocks, threads>>>(n, input_data, output_data);
  return 0;
}

__global__ void relu_gradient_kernel(int n, const float* input, const float* in_grad, float* output){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y>n){
    return;
  }
  output[y] = input[y] > 0 ? in_grad[y] : 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < input->ndim; i++) {
    n *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  relu_gradient_kernel<<<blocks, threads>>>(n, input_data, in_grad_data, output_data);
  return 0;
}

__global__ void matrix_softmax_kernel(int nrow, int ncol, const float *input, float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  output += y * ncol;
  float maxval = *input;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input[x] - maxval);
  }
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval) / sum;
  }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] && input->shape[1] == output->shape[1]);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_kernel<<<1, threads>>>(nrow, ncol, input_data, output_data);
  return 0;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
  const float *input_a,
  const float *input_b,
  float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
  threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
