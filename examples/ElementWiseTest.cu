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

const int SIZE = 512 * 1024 * 1024;
#define loop_size 2

/*
__global__ void template_op_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0) {
	  // [thread_extent] blockIdx.x = 32768
	  // [thread_extent] threadIdx.x = 64
	for (int vthread_s = 0; vthread_s < 64; ++vthread_s) {
		output0[(((((int)blockIdx.x) * 4096) + (vthread_s * 64)) + ((int)threadIdx.x))] = (input0[(((((int)blockIdx.x) * 4096) + (vthread_s * 64)) + ((int)threadIdx.x))] + input1[(((((int)blockIdx.x) * 4096) + (vthread_s * 64)) + ((int)threadIdx.x))]);
	}
}

__global__ void template_op_kernel1( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 65536
	// [thread_extent] threadIdx.x = 64
	for (int vthread_s = 0; vthread_s < 32; ++vthread_s) {
		output0[(((((int)blockIdx.x) * 2048) + (vthread_s * 64)) + ((int)threadIdx.x))] = (input0[(((((int)blockIdx.x) * 2048) + (vthread_s * 64)) + ((int)threadIdx.x))] + input1[(((((int)blockIdx.x) * 2048) + (vthread_s * 64)) + ((int)threadIdx.x))]);
	}
}

__global__ void template_op_kernel2( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0) {
	// [thread_extent] blockIdx.x = 32768
	// [thread_extent] threadIdx.x = 32
	for (int vthread_s = 0; vthread_s < 128; ++vthread_s) {
		output0[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + vthread_s)] = (input0[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + vthread_s)] + input1[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + vthread_s)]);
	}
}

__global__ void template_op_kernel4( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 16384
  // [thread_extent] threadIdx.x = 1024
  output0[((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2))] = (input0[((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2))] + input1[((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2))]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 1)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 1)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 1)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2048)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2048)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2048)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2049)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2049)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 2049)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4096)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4096)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4096)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4097)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4097)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 4097)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6144)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6144)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6144)]);
  output0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6145)] = (input0[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6145)] + input1[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 2)) + 6145)]);
}

__global__ void template_op_kernel5( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 16384
  // [thread_extent] threadIdx.x = 16
  for (int vthread_s = 0; vthread_s < 128; ++vthread_s) {
      output0[(((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4))] = (input0[(((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4))] + input1[(((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4))]);
      output0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 1)] = (input0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 1)] + input1[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 1)]);
      output0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 2)] = (input0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 2)] + input1[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 2)]);
      output0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 3)] = (input0[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 3)] + input1[((((((int)blockIdx.x) * 8192) + (vthread_s * 64)) + (((int)threadIdx.x) * 4)) + 3)]);
  }
}
*/

#include "cu_helper.h"

int main(int argc, char *argv[])
{
	checkCudaErrors(cuInit(0));
	CUdevice device;
	checkCudaErrors(cuDeviceGet(&device, 0));
	CUcontext context;
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

	CUmodule cuModule_;
	CUfunction cuda_func_;
	checkCudaErrors(cuModuleLoad(&cuModule_, "my_kernel.out"));
	checkCudaErrors(cuModuleGetFunction(&cuda_func_, cuModule_, "template_op_kernel0"));

	int TB_count = atoi(argv[1]);
	int TB_size = atoi(argv[2]);
	int iterations = 1;
	int skip = 0;
	float *Ah, *Bh;
	CUdeviceptr Ad, Bd, Cd;
	double *T;
	T = (double*)malloc(iterations * sizeof(double));
	Ah = (float*)malloc(SIZE * sizeof(float));
	Bh = (float*)malloc(SIZE * sizeof(float));
	//Ch = (float*)malloc(SIZE * sizeof(float));

	//cudaMalloc((void **)&Ad, SIZE * sizeof(float));
	//cudaMalloc((void **)&Bd, SIZE * sizeof(float));
	//cudaMalloc((void **)&Cd, SIZE * sizeof(float));

	checkCudaErrors(cuMemAlloc(&Ad, sizeof(float) * SIZE));
	checkCudaErrors(cuMemAlloc(&Bd, sizeof(float) * SIZE));
	checkCudaErrors(cuMemAlloc(&Cd, sizeof(float) * SIZE));

	void* param[] = {&Ad, &Bd, &Cd};

	for (int i = 0; i < SIZE; ++ i)
	{
		Ah[i] = rand();
		Bh[i] = rand();
	}
	checkCudaErrors(cuMemcpyHtoD(Ad, Ah, SIZE * sizeof(float)));
	checkCudaErrors(cuMemcpyHtoD(Bd, Bh, SIZE * sizeof(float)));

	for (int i = 0; i < iterations; ++ i)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		cuLaunchKernel(cuda_func_, TB_count, 1, 1, TB_size, 1, 1, 0, 0, (void**) param, 0);
		//template_op_kernel1 <<<SIZE, 1>>>(Ad, Cd);
		//template_op_kernel2 <<<SIZE, 1>>>(Ad);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::high_resolution_clock::now();
		T[i] = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
	}

	for (int i = 0; i < iterations; ++ i)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		cudaDeviceSynchronize();
		auto t1 = std::chrono::high_resolution_clock::now();
		T[i] -= (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
	}
	
	double sum = 0;
	for (int i = skip; i < iterations; ++ i)
		sum += T[i];
	fprintf(stderr, "avg kernel time: %.2lf ns\n", sum / (iterations - skip));
}
