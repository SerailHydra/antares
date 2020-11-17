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

const int input_size = 4096;
const int output_size = 32 * 1024 * 1024;

#include "cu_helper.h"

__global__ void Broadcast(float* __restrict__ input0, float* __restrict__ output0)
{
	// [thread_extent] blockIdx.x = 1024
	// [thread_extent] threadIdx.x = 4
	// [thread_extent] blockIdx.y = 64
	// [thread_extent] threadIdx.y = 128
	output0[(((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 8192)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)))] = input0[(((((int)blockIdx.x) * 4) + ((int)threadIdx.x)))];
}

__global__ void Broadcast1(float* __restrict__ input0, float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 512
	// [thread_extent] threadIdx.x = 2
	// [thread_extent] blockIdx.y = 64
	// [thread_extent] threadIdx.y = 64
	output0[(((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)))] = input0[(((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 64))] = input0[(((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 8192))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 1))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 8256))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 1))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 16384))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 2))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 16448))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 2))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 24576))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 3))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 24640))] = input0[((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 4)) + 3))];
}

bool check(float *A, float *B)
{
	for (int i = 0; i < 4096; ++ i)
		for (int j = 0; j < 8192; ++ j)
			if (abs(A[i] - B[i * 8192 + j]) > 1e-6)
			{
				fprintf(stderr, "%d %d %f %f\n", i, j, A[i], B[i * 8192 + j]);
				return false;
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
		Ah[i] = rand();

	cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Grid(1024, 64, 1);
	dim3 Block(4, 128, 1);
	for (int i = 0; i < 1; ++ i)
	{
		Broadcast <<<Grid, Block>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (!check(Ah, Bh))
		fprintf(stderr, "error!\n");
	else
		fprintf(stderr, "pass!\n");

        dim3 Grid1(512, 64, 1);
	dim3 Block1(2, 64, 1);
	for (int i = 0; i < 1; ++ i)
	{
		Broadcast1 <<<Grid1, Block1>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (!check(Ah, Bh))
		fprintf(stderr, "error!\n");
	else
		fprintf(stderr, "pass!\n");

}
