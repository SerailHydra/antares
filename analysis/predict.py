# input: ptx, config
# output: the execution time

import os, argparse
import dag
from dag import *

def main(args):
    DAG = dag(args.ptx_file)
    DAG.simulate()
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptx_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The ptx file that contains assembly")
    main(parser.parse_args())
