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

const int input_size0 = 32 * 1024 * 1024;
const int input_size1 = 8192;
const int output_size = 4096;

#include "cu_helper.h"

__global__ void MatVec(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0)
{
	// [thread_extent] blockIdx.x = 128
	float output0_local[1];
	__shared__ float input0_shared[4096];
	__shared__ float input1_shared[128];
	// [thread_extent] threadIdx.x = 32
	output0_local[(0)] = 0.000000e+00f;
	for (int rv_outer_outer = 0; rv_outer_outer < 64; ++rv_outer_outer)
	{
		__syncthreads();
		for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer)
		{
			// [thread_extent] threadIdx.x = 32
			((float4*)(input0_shared + (((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)(input0 + (((((((int)blockIdx.x) * 262144) + (ax0_ax1_fused_outer_outer * 8192)) + (rv_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
		}
		// [thread_extent] threadIdx.x = 32
		((float4*)(input1_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(input1 + (((rv_outer_outer * 128) + (((int)threadIdx.x) * 4)))))[0];
		__syncthreads();
		for (int rv_outer_inner = 0; rv_outer_inner < 128; ++rv_outer_inner) {
			output0_local[(0)] = (output0_local[(0)] + (input0_shared[(((((int)threadIdx.x) * 128) + rv_outer_inner))] * input1_shared[(rv_outer_inner)]));
		}
	}
	output0[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = output0_local[(0)];
}


int main(int argc, char *argv[])
{
	checkCudaErrors(cuInit(0));
	CUdevice device;
	checkCudaErrors(cuDeviceGet(&device, 0));
	CUcontext context;
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

	float *Ah, *Bh;
	float *Ad, *Bd, *Cd;
	Ah = (float*)malloc(input_size0 * sizeof(float));
	Bh = (float*)malloc(input_size1 * sizeof(float));

	cudaMalloc((void **)&Ad, input_size0 * sizeof(float));
	cudaMalloc((void **)&Bd, input_size1 * sizeof(float));
	cudaMalloc((void **)&Cd, output_size * sizeof(float));

	for (int i = 0; i < input_size0; ++ i)
		Ah[i] = rand();
	for (int i = 0; i < input_size1; ++ i)
		Bh[i] = rand();

	cudaMemcpy(Ad, Ah, input_size0 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, Bh, input_size1 * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < 100; ++ i)
	{
		MatVec <<<128, 32>>> (Ad, Bd, Cd);
		cudaDeviceSynchronize();
	}
}
