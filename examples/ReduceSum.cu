///524288/float32/input0,524288/float32/input1:524288/float32/output0
// backend = c-cuda
// CONFIG: 
// COMPUTE_V1: - einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>

const int input_size = 32 * 1024 * 1024;
const int output_size = 4096;

#include "cu_helper.h"

__global__ void ReduceSum(float* __restrict__ input0, float* __restrict__ output0)
{
	// [thread_extent] blockIdx.x = 4096
	float normal_reduce_temp0[1];
	__shared__ float red_buf0[8];
	// [thread_extent] threadIdx.x = 8
	normal_reduce_temp0[(0)] = 0.000000e+00f;
	for (int rv_outer = 0; rv_outer < 1024; ++rv_outer)
		normal_reduce_temp0[(0)] = (normal_reduce_temp0[(0)] + input0[((((((int)blockIdx.x) * 8192) + (rv_outer * 8)) + ((int)threadIdx.x)))]);

	__syncthreads();
	((volatile float*)red_buf0)[(((int)threadIdx.x))] = normal_reduce_temp0[(0)];
	__syncthreads();
	if (((int)threadIdx.x) < 4)
	{
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
	}
	__syncthreads();
	output0[(((int)blockIdx.x))] = ((volatile float*)red_buf0)[(0)];
}

int main(int argc, char *argv[])
{
	checkCudaErrors(cuInit(0));
	CUdevice device;
	checkCudaErrors(cuDeviceGet(&device, 0));
	CUcontext context;
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

	float *Ah;
	float *Ad, *Bd;
	Ah = (float*)malloc(input_size * sizeof(float));

	cudaMalloc((void **)&Ad, input_size * sizeof(float));
	cudaMalloc((void **)&Bd, output_size * sizeof(float));

	for (int i = 0; i < input_size; ++ i)
		Ah[i] = rand();

	cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < 1; ++ i)
	{
		ReduceSum <<<4096, 8>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
}
