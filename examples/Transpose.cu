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

const int input_size = 64 * 1024 * 1024;
const int output_size = input_size;

const int K = 2;
const int stride = 32 / K;

#include "cu_helper.h"

__global__ void Transpose(float *input, float *output)
{
	__shared__ float shared[1024];
	for (int i = 0; i < K; ++ i)
		shared[(threadIdx.y + stride * i) * 32 + (threadIdx.y + threadIdx.x + stride * i) % 32] = input[blockIdx.y * 262144 + blockIdx.x * 32 + (threadIdx.y + stride * i) * 8192 + threadIdx.x];
	__syncthreads();
	for (int i = 0; i < K; ++ i)
		output[blockIdx.x * 262144 + blockIdx.y * 32 + (threadIdx.y + stride * i) * 8192 + threadIdx.x] = shared[threadIdx.x * 32 + (threadIdx.y + threadIdx.x + stride * i) % 32];
}

__global__ void Transpose_(float *input, float *output)
{
	__shared__ float shared[1024];
	for (int i = 0; i < K; ++ i)
		shared[threadIdx.y * 32 + (threadIdx.x + stride * i)] = input[blockIdx.y * 262144 + blockIdx.x * 32 + threadIdx.y * 8192 + threadIdx.x + stride * i];
	__syncthreads();
	for (int i = 0; i < K; ++ i)
		output[blockIdx.x * 262144 + blockIdx.y * 32 + threadIdx.y * 8192 + threadIdx.x + stride * i] = shared[(threadIdx.x + stride * i) * 32 + threadIdx.y];
}

__global__ void Transpose1(float* __restrict__ input0, float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 1024
	// [thread_extent] threadIdx.x = 4
	// [thread_extent] blockIdx.y = 64
	// [thread_extent] threadIdx.y = 128
	output0[(((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 16384)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)))] = input0[(((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)))];
	output0[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 16384)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.y)) + 8192))] = input0[((((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + 1))];
}

__global__ void Transpose2(float* __restrict__ input0, float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 2048
	// [thread_extent] threadIdx.x = 4
	// [thread_extent] blockIdx.y = 128
	// [thread_extent] threadIdx.y = 32
	output0[((((((int)blockIdx.x) * 32768) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y))) + threadIdx.x * 8192] = input0[((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 4))) + threadIdx.x];
	output0[(((((((int)blockIdx.x) * 32768) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + threadIdx.x * 8192 + 32))] = input0[(((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 4)) + 262144)) + threadIdx.x];
}

__global__ void Transpose3(float* __restrict__ input0, float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 1024
	// [thread_extent] threadIdx.x = 4
	// [thread_extent] blockIdx.y = 128
	// [thread_extent] threadIdx.y = 64
	output0[(((((int)blockIdx.x) * 65536) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + threadIdx.x * 8192] = input0[((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 8))) + threadIdx.x];
	output0[(((((int)blockIdx.x) * 65536) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + (threadIdx.x + 4) * 8192] = input0[((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 8))) + threadIdx.x + 4];
}

bool check(float *a, float *b)
{
	bool ret = true;
	for (int i = 0; i < 8192; ++ i)
		for (int j = 0; j < 8192; ++ j)
			if (abs(a[i * 8192 + j] - b[i + j * 8192]) > 1e-6)
				ret = false;
	return ret;
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
		Ah[i] = i;
	cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 Grid(256, 256, 1);
	dim3 Block(32, stride, 1);
	for (int i = 0; i < 100; ++ i)
	{
		Transpose <<<Grid, Block>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (check(Ah, Bh))
		fprintf(stderr, "pass!\n");
	else
		fprintf(stderr, "error!\n");

	dim3 Grid1(1024, 64, 1);
	dim3 Block1(4, 128, 1);
	for (int i = 0; i < 1; ++ i)
	{
		Transpose1 <<<Grid1, Block1>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (check(Ah, Bh))
		fprintf(stderr, "pass!\n");
	else
		fprintf(stderr, "error!\n");
	
        dim3 Grid2(2048, 128, 1);
	dim3 Block2(4, 32, 1);
	for (int i = 0; i < 100; ++ i)
	{
		Transpose2 <<<Grid2, Block2>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (check(Ah, Bh))
		fprintf(stderr, "pass!\n");
	else
		fprintf(stderr, "error!\n");
	
        dim3 Grid3(1024, 128, 1);
	dim3 Block3(4, 64, 1);
	for (int i = 0; i < 100; ++ i)
	{
		Transpose3 <<<Grid3, Block3>>> (Ad, Bd);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(Bh, Bd, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (check(Ah, Bh))
		fprintf(stderr, "pass!\n");
	else
		fprintf(stderr, "error!\n");
}
