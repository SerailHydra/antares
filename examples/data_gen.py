import os, torch
import json
from torch.contrib.antares.custom_op import CustomOp

sample_dir = "samples"

benchmarks = {
    "Broadcast": '- einstein_v2("output0[N, F, HO, WO] = input0[N] where F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [16]}})',
    "BroadcastAll": '- einstein_v2("output0[N, F, HO, WO] = input0[0] where N in 8, F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [1]}})',
    "MatMul": '- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M]", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})',
    "BatchMatMul": '- einstein_v2("output0[B, N, M] +=! input0[B, N, K] * input1[B, K, M]", input_dict={"input0": {"dtype": "float32", "shape": [3, 1024, 512]}, "input1": {"dtype": "float32", "shape": [3, 512, 512]}})',
    "ElementWise": '- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 1024 * 128]}, "input1": {"dtype": "float32", "shape": [1024 * 1024 * 128]}})',
    "Transpose": '- einstein_v2("output0[N, M] = input0[M, N]", input_dict={"input0": {"dtype": "float32", "shape": [8 * 1024, 8 * 1024]}})',
    "Reduce": '- einstein_v2("output0[A, B, C] = input0[A, B, C // 64, C % 64] where C in 128", input_dict={"input0": {"dtype": "float32", "shape": [3, 3, 2, 64]}})',
    "ReduceSum": '- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 64, 2 * 1024]}})',
    "ReduceMin": '- einstein_v2("output0[N] <=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})',
    "ConditionalRelu": '- einstein_v2("output0[N] = input0[N].when([input0[N] > 0.0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}})',
    "ConvolutionNoPad": '- _N, _C, _HW, _F, _K, _S = 512, 3, 227, 64, 11, 4; _HWO = (_HW - _K) // _S + 1; einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * %d + KH, WO * %d + KW] * input1[F, C, KH, KW] where HO in %d, WO in %d" % (_S, _S, _HWO, _HWO), { "input0": {"dtype": "float32", "shape": [_N, _C, _HW, _HW]}, "input1": {"dtype": "float32", "shape": [_F, _C, _K, _K]}})',
    "ConvolutionWithPad": '- _N, _C, _HW, _F, _K, _S, _P = 64, 64, 27, 192, 5, 1, 2; _HWO = (_HW - _K + _P * 2) // _S + 1; einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * %d + KH - %d, WO * %d + KW - %d].when([HO * %d + KH - %d >= 0, HO * %d + KH - %d < %d, WO * %d + KW - %d >= 0, WO * %d + KW - %d < %d], 0.0) * input1[F, C, KH, KW] where HO in %d, WO in %d" % (_S, _P, _S, _P, _S, _P, _S, _P, _HW, _S, _P, _S, _P, _HW, _HWO, _HWO), { "input0": {"dtype": "float32", "shape": [_N, _C, _HW, _HW]}, "input1": {"dtype": "float32", "shape": [_F, _C, _K, _K]}})',
    "DepthToSpace": '- einstein_v2("output0[N, H, C0, W, C1, C2] = input0[N, H, W, C0, C1, C2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 256, 256, 2, 2, 4]}})',
    "DepthwiseConv": '- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [3, 3, 16]}})',
    "Slice": '- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})',
    "Concat": '- einstein_v2("output0[N, F] = input0[N, F].when([F < 128], input1[N, F - 128]) where F in 256", input_dict={"input0": {"dtype": "float32", "shape": [4, 128]}, "input1": {"dtype": "float32", "shape": [4, 128]}})',
    "OneHot": '- einstein_v2("output0[N, F] = parse(1.0).when([input0[N] == F], 0.0) where F in 128", input_dict={"input0": {"dtype": "int32", "shape": [4]}})',
    "Take": '- einstein_v2("output0[F, C] = input0[input1[F], C]", input_dict={"input0": {"dtype": "float32", "shape": [30528, 1024]}, "input1": {"dtype": "int32", "shape": [3072]}})',
    "Gather": '- einstein_v2("output0[N, F] = input0[input1[N, F]]", input_dict={"input0": {"dtype": "float32", "shape": [65536]}, "input1": {"dtype": "int32", "shape": [4, 64]}})',
    "Pad": '- einstein_v2("output0[N, C, HO, WO] = input0[N, C, -1 + HO, -1 + WO].when([-1 + HO >= 0, -1 + HO < 32, -1 + WO >= 0, -1 + WO < 32], 0.0) where HO in 34, WO in 34", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 32, 32]}})',
    "DivNoNan": '- einstein_v2("output0[N] = (input0[N] / input1[N]).when([input1[N] != 0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [32 * 1024]}, "input1": {"dtype": "float32", "shape": [32 * 1024]}})',
    "MaxPool": '- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})',
    "AvgPool": '- einstein_v2("temp0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] where HO in 85, WO in 85, KW in 3, KH in 3; output0[NC, HO, WO] = temp0[NC, HO, WO] * 0.111111", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})',
    "Tile": '- einstein_v2("output0[ON, OC] = input0[ON % 2, OC % 16] where ON in 1024, OC in 4096", input_dict={"input0": {"dtype": "float32", "shape": [2, 16]}})',
    "SoftmaxV1": '- einstein_v2("temp0[N] >=! input0[N, C]",                                            { "input0": {"dtype": "float32", "shape": [32, 1024]} })',
    "SoftmaxV2": '- einstein_v2("temp1[N] +=! (input0[N, C] - temp0[N]).call(\"exp\")",                 { "input0": {"dtype": "float32", "shape": [32, 1024]}, "temp0": {"dtype": "float32", "shape": [32]} })',
    "SoftmaxV3": '- einstein_v2("output0[N, C] = (input0[N, C] - temp0[N]).call(\"exp\") / temp1[N]",   { "input0": {"dtype": "float32", "shape": [32, 1024]}, "temp0": {"dtype": "float32", "shape": [32]}, "temp1": {"dtype": "float32", "shape": [32]} })'
}

configs = {
        "ElementWise": ['{"axis_0": [-1, 16, 64, 1], "reorder": [0]}', '{"axis_0": [-1, 4, 1024, 2], "reorder": [0]}', '{"axis_0": [-1, 64, 64, 1], "reorder": [0]}', '{"axis_0": [-1, 32, 64, 1], "reorder": [0]}', '{"axis_0": [-1, 1, 32, 128], "reorder": [0]}', '{"axis_0": [-1, 128, 16, 4], "reorder": [0]}'],
        "ReduceSum": ['{"axis_0": [-1, 4, 32, 8], "reorder": [0], "reduce_0": [-1, 8, 16]}', '{"axis_0": [-1, 4, 64, 8], "reorder": [0], "reduce_0": [-1, 4, 32]}', '{"axis_0": [-1, 4, 64, 8], "reorder": [0], "reduce_0": [-1, 16, 4]}', '{"axis_0": [-1, 4, 64, 8], "reorder": [0], "reduce_0": [-1, 32, 1]}']
}

def write_config(config, operator, dir_path, args, test_prog):
    os.system("cd ..")
    os.system("sudo BACKEND=c-cuda CONFIG=\'{}\' COMPUTE_V1=\'{}\' make".format(config, operator))
    os.system("cd examples")
    os.makedirs(dir_path)
    os.system("sudo mv /mydata/libAntares/cache/* {}".format(dir_path))
    os.system("cp {}/_/my_kernel.out .".format(dir_path))
    os.system("/usr/local/cuda/bin/nvcc -lcuda {}.cu -o {}".format(test_prog, test_prog))
    os.system("sudo /usr/local/cuda/bin/nvprof --metrics dram_read_bytes,dram_write_bytes ./{} {}".format(test_prog, args))
    os.system("sudo /opt/nvidia/nsight-compute/2020.1.2/ncu --set full ./{} {}".format(test_prog, args))


clean = False
if clean:
    if os.path.exists(sample_dir):
        os.system("rm -rf {}".format(sample_dir))
    os.makedirs(sample_dir)

for key in benchmarks:
    if key != "ReduceSum":
        continue
    #if key in configs:
    
    v = 0
    """
    # ElementWise configs
    for i in range(0, 9):
        for j in range(0, 9):
            for k in range(0, 9):
                #for config in configs[key]:
                config = '{' + "\"axis_0\"" + ': [-1, {}, {}, {}], "reorder": [0]'.format(pow(2, i), pow(2, j), pow(2, k)) + '}'
                cmd = "cd ..; sudo BACKEND=c-cuda CONFIG=\'{}\' COMPUTE_V1=\'{}\' make; cd examples".format(config, benchmarks[key])
                print(cmd)
                dir_path = os.path.join(sample_dir, key, str(v))
                os.makedirs(dir_path)
                os.system(cmd)
                dim = json.loads(config)["axis_0"]
                size = 128 * 1024 * 1024
                TB_count = size // (dim[1] * dim[2] * dim[3])
                TB_size = dim[2]
                os.system("sudo mv /mydata/libAntares/cache/* {}".format(dir_path))
                os.system("cp {}/_/my_kernel.out .".format(dir_path))
                os.system("/usr/local/cuda/bin/nvcc -lcuda ElementWiseTest.cu -o ElementWiseTest")
                os.system("sudo /usr/local/cuda/bin/nvprof --metrics dram_read_bytes,dram_write_bytes ./ElementWiseTest {} {}".format(TB_count, TB_size))
                os.system("sudo /opt/nvidia/nsight-compute/2020.1.2/ncu --set full ./ElementWiseTest {} {}".format(TB_count, TB_size))
                v += 1
    """

    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                size = 64 * 1024
                dim = [-1, pow(2, i), pow(2, j), pow(2, k)]
                TB_count = size // (dim[1] * dim[2] * dim[3])
                if TB_count < 1:
                    continue
                #for config in configs[key]:
                config = '{' + "\"axis_0\"" + ': [-1, {}, {}, {}], "reorder": [0], "reduce_0": [-1, 1, 1]'.format(pow(2, i), pow(2, j), pow(2, k)) + '}'
                cmd = "cd ..; sudo BACKEND=c-cuda CONFIG=\'{}\' COMPUTE_V1=\'{}\' make; cd examples".format(config, benchmarks[key])
                print(cmd)
                dir_path = os.path.join(sample_dir, key, str(v))
                os.makedirs(dir_path)
                os.system(cmd)
                TB_size = dim[2]
                os.system("sudo mv /mydata/libAntares/cache/* {}".format(dir_path))
                os.system("cp {}/_/my_kernel.out .".format(dir_path))
                os.system("/usr/local/cuda/bin/nvcc -lcuda ReduceSumTest.cu -o ReduceSumTest")
                os.system("sudo /usr/local/cuda/bin/nvprof --metrics dram_read_bytes,dram_write_bytes ./ReduceSumTest {} {}".format(TB_count, TB_size))
                os.system("sudo /opt/nvidia/nsight-compute/2020.1.2/ncu --set full ./ReduceSumTest {} {}".format(TB_count, TB_size))
                v += 1

    """
    for config in configs[key]:
        cmd = "cd ..; sudo CONFIG=\'{}\' BACKEND=c-cuda COMPUTE_V1=\'".format(config) + benchmarks[key] + "\' make; cd examples"
        data_path = os.path.join(sample_dir, key, str(v))
        os.system("rm -rf {}".format(data_path))
        os.makedirs(data_path)
        print(cmd)
        os.system(cmd)
        os.system("sudo mv /mydata/libAntares/cache/* {}".format(data_path))
        v += 1
    """
