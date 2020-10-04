# input: ptx, config
# output: the execution time

import os, argparse
import dag
from dag import *

def estimate_TB(ptx_parser, hw_config)
    TB_perf["pipeline_latency"] = 
    TB_perf["transfer_latency"] = 
    TB_perf["compute_latency"] = 
    return TB_perf

def estimate_batch(ptx_parser, hw_config, batch_size, TB_perf):
    # Estimate the given the performance of one TB
    pass

def estimate_all(source_parser, ptx_parser, hw_config):
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
    
    # step 1: estimating the performance of a single TB
    TB_perf = estimate_TB(ptx_parser, hw_config)
    
    # step 2: estimating the performance of a TB batch
    TB_size = # how many threads in one TB
    batch_size = # how many TBs can execute in parallel

    TB_count = ... # number of TBs
    batch_count = TB_count / batch_size # number of TB batches
    batch_latency = estimate_batch(ptx_parser, hw_config, batch_size, TB_perf)

    rem_size = TB_count - batch_count * batch_size
    rem_latency = estimate_batch(ptx_parser, hw_config, rem_size, TB_perf)
    
    # step 3: estimating the final performance
    return batch_latency * batch_count + rem_latency

def main(args):
    ptx_p = ptx_parser(args.ptx_file)
    pass


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
