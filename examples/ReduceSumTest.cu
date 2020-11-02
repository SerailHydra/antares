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

#include "cu_helper.h"

const int input_size = 64 * 1024 * 2 * 1024;
const int output_size = 64 * 1024;

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

	float *h_input;
	CUdeviceptr d_input, d_output;
	
	h_input = (float*)malloc(input_size * sizeof(float));
	checkCudaErrors(cuMemAlloc(&d_input, input_size * sizeof(float)));
	for (int i = 0; i < input_size; ++ i)
		h_input[i] = rand();
	checkCudaErrors(cuMemcpyHtoD(d_input, h_input, input_size * sizeof(float)));
	checkCudaErrors(cuMemAlloc(&d_output, output_size * sizeof(float)));

	void *param[] = {&d_input, &d_output};

	auto t0 = std::chrono::high_resolution_clock::now();
	checkCudaErrors(cuLaunchKernel(cuda_func_, TB_count, 1, 1, TB_size, 1, 1, 0, 0, (void**) param, 0));
	//template_op_kernel1 <<<SIZE, 1>>>(Ad, Cd);
	//template_op_kernel2 <<<SIZE, 1>>>(Ad);
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	double duration = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
	fprintf(stderr, "duration: %lf\n", duration);
}
