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
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

const int input_size0 = 64 * 1024 * 1024;
const int input_size1 = 64 * 1024 * 1024;
const int output_size = 64 * 1024 * 1024;

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
	//checkCudaErrors(cuModuleLoad(&cuModule_, "my_kernel.out"));
	//checkCudaErrors(cuModuleGetFunction(&cuda_func_, cuModule_, "template_op_kernel0"));

	int id = atoi(argv[1]);
	//int TB_size = atoi(argv[2]);
	float *Ah, *Bh;
	CUdeviceptr Ad, Bd, Cd;
	Ah = (float*)malloc(input_size0 * sizeof(float));
	Bh = (float*)malloc(input_size1 * sizeof(float));
	//Ch = (float*)malloc(SIZE * sizeof(float));

	//cudaMalloc((void **)&Ad, SIZE * sizeof(float));
	//cudaMalloc((void **)&Bd, SIZE * sizeof(float));
	//cudaMalloc((void **)&Cd, SIZE * sizeof(float));

	checkCudaErrors(cuMemAlloc(&Ad, sizeof(float) * input_size0));
	checkCudaErrors(cuMemAlloc(&Bd, sizeof(float) * input_size1));
	checkCudaErrors(cuMemAlloc(&Cd, sizeof(float) * output_size));

	void* param[] = {&Ad, &Bd, &Cd};

	for (int i = 0; i < input_size0; ++ i)
		Ah[i] = rand();
	for (int i = 0; i < input_size1; ++ i)
		Bh[i] = rand();

	checkCudaErrors(cuMemcpyHtoD(Ad, Ah, input_size0 * sizeof(float)));
	checkCudaErrors(cuMemcpyHtoD(Bd, Bh, input_size1 * sizeof(float)));

	std::string path = "results/MatMul/" + std::to_string(id);
	std::string code_path = path + "/my_kernel.cc";
	std::string mod_path = path + "/my_kernel.out";
	checkCudaErrors(cuModuleLoad(&cuModule_, mod_path.c_str()));
	checkCudaErrors(cuModuleGetFunction(&cuda_func_, cuModule_, "template_op_kernel0"));
	auto fp = fopen(code_path.c_str(), "r");
	int TB_size, TB_count;
	while (!feof(fp))
	{
		char *line;
		line = (char*)malloc(1000 * sizeof(char));
	       	fgets(line, 1000, fp);
		std::string std_line = std::string(line);
		if (int(std_line.find("[thread_extent] blockIdx.x")) > -1)
		{
			int k = std_line.rfind("=");
			TB_count = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
		}
		if (int(std_line.find("[thread_extent] threadIdx.x")) > -1)
		{
			int k = std_line.rfind("=");
			TB_size = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
		}
	}
	for (int i = 0; i < 1; ++ i)
	{
		checkCudaErrors(cuLaunchKernel(cuda_func_, TB_count, 1, 1, TB_size, 1, 1, 0, 0, (void**) param, 0));
		cudaDeviceSynchronize();
	}

}

