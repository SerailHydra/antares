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
#include <algorithm>

const int input_size = 32 * 1024 * 1024;
const int output_size = 4096;

#include "cu_helper.h"

__global__ void ReduceSum_(float* __restrict__ input0, float* __restrict__ output0)
{
	// [thread_extent] blockIdx.x = 4096
	float sum = 0;
	__shared__ float red_buf0[32];
	// [thread_extent] threadIdx.x = 8
	for (int rv_outer = 0; rv_outer < 256; ++rv_outer)
		sum = sum + input0[((((((int)blockIdx.x) * 8192) + (rv_outer * 32)) + ((int)threadIdx.x)))];

	__syncthreads();
	((volatile float*)red_buf0)[(((int)threadIdx.x))] = sum;
	__syncthreads();
	if (((int)threadIdx.x) < 16)
	{
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
		((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
	}
	__syncthreads();
	output0[(((int)blockIdx.x))] = ((volatile float*)red_buf0)[(0)];
}

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
	output0[(((int)blockIdx.x))] = ((volatile float*)red_buf0)[(0)];
}

__global__ void ReduceSum_antares(float* __restrict__ input0, float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 256
	float output0_rf[1];
	__shared__ float red_buf0[128];
	// [thread_extent] threadIdx.y = 16
	// [thread_extent] threadIdx.x = 8
	output0_rf[(0)] = 0.000000e+00f;
	for (int rv_outer = 0; rv_outer < 1024; ++rv_outer) {
		output0_rf[(0)] = (output0_rf[(0)] + input0[(((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 8192)) + (rv_outer * 8)) + ((int)threadIdx.x)))]);
	}
	__syncthreads();
	((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = output0_rf[(0)];
	__syncthreads();
	if (((int)threadIdx.x) < 4) {
		((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 4))]);
	        ((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 2))]);
		((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = (((volatile float*)red_buf0)[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] + ((volatile float*)red_buf0)[((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 1))]);
	}
	__syncthreads();
	if (((int)threadIdx.x) == 0) {
		output0[(((((int)blockIdx.x) * 16) + ((int)threadIdx.y)))] = ((volatile float*)red_buf0)[((((int)threadIdx.y) * 8))];
	}
}

bool check(float *A, float *B)
{
	for (int i = 0; i < 4096; ++ i)
	{
		double sum = 0;
		for (int j = 0; j < 8192; ++ j)
			sum += A[i * 8192 + j];
		if (abs(sum - B[i]) > 1e-6)
		{
			fprintf(stderr, "%d %f %f\n", i, B[i], sum);
			return false;
		}
	}
	return true;
}

int main(int argc, char *argv[])
{
	checkCudaErrors(cuInit(0));
	CUdevice device;
	checkCudaErrors(cuDeviceGet(&device, 0));
	CUcontext context;
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

	float *Ah, *Bh;
	float *Ad, *Bd;
	Ah = (float*)malloc(input_size * sizeof(float));
	Bh = (float*)malloc(output_size * sizeof(float));

	cudaMalloc((void **)&Ad, input_size * sizeof(float));
	cudaMalloc((void **)&Bd, output_size * sizeof(float));

	for (int i = 0; i < input_size; ++ i)
		Ah[i] = i % 1990;

	cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < 1; ++ i)
	{
		//ReduceSum <<<4096, 8>>> (Ad, Bd);
		ReduceSum_ <<<4096, 32>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);

	if (check(Ah, Bh))
		fprintf(stderr, "pass!\n");
	else
		fprintf(stderr, "error!\n");

	dim3 Grid(256, 1, 1);
	dim3 Block(8, 16, 1);
	for (int i = 0; i < 1; ++ i)
	{
		//ReduceSum <<<4096, 8>>> (Ad, Bd);
		ReduceSum_antares <<<Grid, Block>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
}
