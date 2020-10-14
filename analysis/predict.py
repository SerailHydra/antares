# input: ptx, config
# output: the execution time

import os, argparse
import ptx_parser, gpu_config
from ptx_parser import *
from gpu_config import *

def get_TB_size_count(cu_file):
    # return the # of threads in one TB and # of TBs
    # this information is written in the .cu source file, with lines that contain "[thread_extent]"
    
    TB_count_x = 1
    TB_count_y = 1
    TB_size_x = 1
    TB_size_y = 1
    with open(cu_file, "r") as f:
        for line in f.readlines():
            if not "[thread_extent]" in line:
                continue
            val = int(line.split(" ")[-1])
            if "blockIdx.x" in line:
                TB_count_x = val
            if "blockIdx.y" in line:
                TB_count_x = val
            if "threadIdx.x" in line:
                TB_size_x = val
            if "threadIdx.y" in line:
                TB_size_x = val
    return TB_size_x * TB_size_y, TB_count_x * TB_count_y

def estimate_TB(ptx_parser, hw_config, batch_size):
    # Estimate the given the performance of one TB 
    # need to know how many TBs can run concurrently (batch_size) to determine the DRAM traffic
    TB_perf = {}
    TB_perf["pipeline_latency"] = 0
    TB_perf["transfer_latency"] = batch_size * 8 / hw_config.DRAM_BW
    TB_perf["compute_latency"] = 0
    return TB_perf

def estimate_batch(ptx_parser, hw_config, batch_size, TB_perf):
    return TB_perf["transfer_latency"]
    # Estimate the given the performance of one TB batch, given the performance of single TBs

def estimate_all(cu_file, ptx_parser, hw_config):
    # The process of estimating performance
    # input:
    #   ptx_parser: object containing all information related to operators
    #   hw_config: object containing all information related to hardware
    # output:
    #   The prediction of the execution time
    #
    # The estimation process has three steps:
    # 1, Estimate the pipeline latency, transfer latency and compute latency of a single thread block (TB)
    # 2, Estimate the performance of one TB batch (all TBs that can execute in parallel)
    # 3, Estimate the total elapsed time
    
    TB_size, TB_count = get_TB_size_count(cu_file)
    # how many TBs can execute in parallel (size of one TB batch)
    batch_size = min(TB_count,
                     hw_config.RF_SIZE * 1024 // ptx_parser.register_size_used())
    print("reg size {}".format(ptx_parser.register_size_used()))
    print("size of a TB batch is {}".format(batch_size))
    if ptx_parser.shared_memory_used() != 0:
        batch_size = min(batch_size,
                         hw_config.SHARED_MEMORY_SIZE * 1024 // ptx_parser.shared_memory_used())
    batch_count = TB_count // batch_size # number of TB batches
    rem_size = TB_count - batch_count * batch_size

    # step 1: estimating the performance of a single TB
    TB_perf_batch = estimate_TB(ptx_parser, hw_config, batch_size)
    TB_perf_rem = estimate_TB(ptx_parser, hw_config, rem_size)
    
    # step 2: estimating the performance of a TB batch
    batch_latency = estimate_batch(ptx_parser, hw_config, batch_size, TB_perf_batch)
    rem_latency = estimate_batch(ptx_parser, hw_config, rem_size, TB_perf_rem)
    
    # step 3: estimating the final performance
    return batch_latency * batch_count + rem_latency

def main(args):
    ptx_ = ptx_parser(args.ptx_file)
    cu_ = args.cu_file
    print(estimate_all(cu_, ptx_, GPU_Config()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cu_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The file that contains compiled CUDA code")
    parser.add_argument("--ptx_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The ptx file that contains assembly")
    main(parser.parse_args())
